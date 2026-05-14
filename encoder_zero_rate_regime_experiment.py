"""
Encoder比较：LSTM vs TCN vs Transformer
测试不同零值率（25%, 50%, 75%）对Regime学习的影响
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────
# 1. 生成数据（控制零值率）
# ─────────────────────────────────────────

def generate_data(n_asins=30, n_weeks=104, target_zero_rate=0.75):
    """
    target_zero_rate：目标零值率
    0.25 → 25%是零（需求频繁）
    0.50 → 50%是零（中等稀疏）
    0.75 → 75%是零（高度稀疏）
    """
    rows = []

    # 根据目标零值率反推base_rate
    # zero_rate ≈ 1 - base_rate（简化）
    # base_rate控制购买概率的基础水平
    base_rate_scale = 1 - target_zero_rate

    for asin_id in range(n_asins):
        season_strength = np.random.uniform(0.2, 0.8)
        # 调整base_rate使零值率接近目标
        base_rate = np.random.uniform(
            base_rate_scale * 0.5,
            base_rate_scale * 1.0
        )
        peak_week = np.random.randint(0, 52)

        for week in range(n_weeks):
            season = np.sin(2 * np.pi * (week - peak_week) / 52)
            p_buy  = base_rate + season_strength * max(0, season)
            p_buy  = np.clip(p_buy, 0.01, 0.95)

            if np.random.rand() < p_buy:
                mu     = 3 + 5 * max(0, season)
                demand = np.random.negative_binomial(1, 1/(1+mu))
                demand = min(demand, 50)
            else:
                demand = 0

            rows.append({
                'ASIN':   f'ASIN_{asin_id:03d}',
                'Week':   pd.Timestamp('2023-10-01') +
                          pd.Timedelta(weeks=week),
                'Demand': int(demand)
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 2. 特征工程
# ─────────────────────────────────────────

def make_features(df):
    df = df.copy()
    df['Week'] = pd.to_datetime(df['Week'])
    df = df.sort_values(['ASIN', 'Week']).reset_index(drop=True)
    min_week = df['Week'].min()
    df['t']  = ((df['Week'] - min_week).dt.days // 7).astype(int)

    data = {}
    for asin, group in df.groupby('ASIN'):
        group  = group.reset_index(drop=True)
        demand = group['Demand'].values.astype(float)
        t      = group['t'].values
        T      = len(demand)

        v_t = np.log1p(demand)
        b_t = (demand > 0).astype(float)

        d_t  = np.zeros(T)
        last = -1
        for i in range(T):
            if b_t[i] > 0:
                last = i
            d_t[i] = (i - last) / 52.0 if last >= 0 else 1.0

        sin_year = np.sin(2 * np.pi * t / 52)
        cos_year = np.cos(2 * np.pi * t / 52)

        features = np.stack(
            [v_t, b_t, d_t, sin_year, cos_year],
            axis=1
        ).astype(np.float32)

        data[asin] = {
            'features': features,
            'demand':   demand.astype(np.float32)
        }

    return data


# ─────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────

class DemandDataset(Dataset):
    def __init__(self, data, history=52, horizon=20):
        self.samples = []
        for asin, d in data.items():
            features = d['features']
            demand   = d['demand']
            T = len(demand)

            # 计算这个ASIN的历史基准均值
            hist_mean = demand[:history].mean() if history <= T else demand.mean()
            hist_mean = max(hist_mean, 0.01)  # 避免除零

            for start in range(T - history - horizon + 1):
                x      = features[start:start+history]
                y      = demand[start+history:start+history+horizon]

                # 最近4周的需求水平
                recent = demand[start+history-4:start+history].mean()

                # 相对于历史均值判断regime
                # 高于历史均值 → 激活，低于 → 休眠
                regime = 1 if recent > hist_mean * 0.8 else 0

                self.samples.append({
                    'x':      torch.tensor(x),
                    'y':      torch.tensor(y),
                    'regime': regime
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────
# 4. 三种Encoder
# ─────────────────────────────────────────

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=5, d_model=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.norm(h[-1])

    def name(self): return "LSTM"


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv    = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            dilation=dilation
        )

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNEncoder(nn.Module):
    def __init__(self, input_dim=5, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # 短期精细 + 长期覆盖
        dilations = [1, 2, 3, 4, 8, 26, 52]
        self.convs = nn.ModuleList([
            CausalConv1d(d_model, d_model, kernel_size=2, dilation=d)
            for d in dilations
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in dilations
        ])

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x)
            x = x + residual
            x = x.permute(0, 2, 1)
            x = norm(x)
            x = x.permute(0, 2, 1)
            x = F.gelu(x)
        return x[:, :, -1]

    def name(self): return "TCN"


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=5, d_model=64,
                 n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x  = self.input_proj(x)
        x  = self.transformer(x)
        return self.norm(x[:, -1])


    def name(self): return "Transformer"


# ─────────────────────────────────────────
# 4d. TCN + Sparse Attention
# 同时学习：零值模式（TCN）+ 非零事件关系（SparseAttn）
# ─────────────────────────────────────────

class SparseAttention(nn.Module):
    """
    只在非零位置做attention
    → 专注学习非零事件之间的关系
    """
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, b_t):
        """
        x:   [B, T, d_model]  TCN的输出
        b_t: [B, T]           哪些位置有购买（非零mask）
        """
        B, T, _ = x.shape

        # key_padding_mask：True的位置被忽略
        # 我们忽略零值位置（b_t==0的位置）
        attn_mask = (b_t == 0)  # [B, T]

        # 如果某个样本全是零 → 退化到全attention
        all_zero = attn_mask.all(dim=1, keepdim=True)  # [B, 1]
        attn_mask = attn_mask & ~all_zero              # [B, T]

        out, _ = self.attn(
            x, x, x,
            key_padding_mask=attn_mask
        )

        return self.norm(x + out)  # 残差连接


class TCNSparseAttnEncoder(nn.Module):
    """
    TCN：学习零值模式（休眠/激活信号）
    SparseAttn：学习非零事件关系（购买模式）

    同时学习0和非0
    → h_t包含最完整的regime信息
    """
    def __init__(self, input_dim=5, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        # TCN：学习零值模式
        dilations = [1, 2, 3, 4, 8, 26, 52]
        self.convs = nn.ModuleList([
            CausalConv1d(d_model, d_model,
                        kernel_size=2, dilation=d)
            for d in dilations
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in dilations
        ])

        # Sparse Attention：学习非零事件关系
        self.sparse_attn = SparseAttention(d_model, n_heads=4)

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, T, input_dim]
        """
        # b_t：哪些位置有购买（第二个channel）
        b_t = x[:, :, 1]  # [B, T]

        # Step 1：TCN扫描零值模式（序列输出）
        h = self.input_proj(x)      # [B, T, d_model]
        h = h.permute(0, 2, 1)     # [B, d_model, T]

        for conv, norm in zip(self.convs, self.norms):
            residual = h
            h = conv(h)
            h = h + residual
            h = h.permute(0, 2, 1)
            h = norm(h)
            h = h.permute(0, 2, 1)
            h = F.gelu(h)

        h = h.permute(0, 2, 1)     # [B, T, d_model]

        # Step 2：Sparse Attention学习非零事件关系
        h = self.sparse_attn(h, b_t)  # [B, T, d_model]

        # 取最后时间步
        h_t = self.final_norm(h[:, -1, :])  # [B, d_model]

        return h_t

    def name(self): return "TCN+SparseAttn"


# ─────────────────────────────────────────
# 4b. TCN + Sparse Attention（新增）
# 同时学习零值模式和非零事件关系
# ─────────────────────────────────────────

class SparseAttention(nn.Module):
    """
    只在非零位置做attention
    学习非零事件之间的关系
    """
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, b_t):
        """
        x:   [B, T, d_model]  TCN的输出
        b_t: [B, T]           哪些位置有购买
        """
        # key_padding_mask: True的位置被忽略
        # 零值位置 = True（忽略）
        # 非零位置 = False（保留）
        mask = (b_t == 0)  # [B, T]

        # 如果某个样本全是零，退化到全attention
        all_zero = mask.all(dim=1, keepdim=True)  # [B, 1]
        mask = mask & ~all_zero  # 全零样本不mask

        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return self.norm(x + out)  # 残差连接


class TCNSparseAttnEncoder(nn.Module):
    """
    TCN：学习零值模式（学0）
    Sparse Attention：学习非零事件关系（学非0）
    两者合在一起：完整的regime信息
    """
    def __init__(self, input_dim=5, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        # TCN部分：扫描零值模式
        dilations = [1, 2, 3, 4, 8, 26, 52]
        self.convs = nn.ModuleList([
            CausalConv1d(d_model, d_model,
                        kernel_size=2, dilation=d)
            for d in dilations
        ])
        self.tcn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in dilations
        ])

        # Sparse Attention部分：学习非零事件关系
        self.sparse_attn = SparseAttention(d_model, n_heads=4)

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, input_dim]
        # b_t：哪些位置有购买（第二个channel）
        b_t = x[:, :, 1]  # [B, T]

        # Step 1：TCN扫描零值模式
        h = self.input_proj(x)    # [B, T, d_model]
        h = h.permute(0, 2, 1)   # [B, d_model, T]

        for conv, norm in zip(self.convs, self.tcn_norms):
            residual = h
            h = conv(h)
            h = h + residual
            h = h.permute(0, 2, 1)
            h = norm(h)
            h = h.permute(0, 2, 1)
            h = F.gelu(h)

        h = h.permute(0, 2, 1)   # [B, T, d_model]

        # Step 2：Sparse Attention学习非零事件关系
        h = self.sparse_attn(h, b_t)

        # 取最后时间步
        h_t = self.final_norm(h[:, -1, :])  # [B, d_model]
        return h_t

    def name(self): return 'TCN+SparseAttn'


# ─────────────────────────────────────────
# 5. 预测头
# ─────────────────────────────────────────

class ForecastHead(nn.Module):
    def __init__(self, d_model=64, horizon=20):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, horizon)
        )

    def forward(self, h_t):
        return F.softplus(self.head(h_t))


class Forecaster(nn.Module):
    def __init__(self, encoder, d_model=64, horizon=20):
        super().__init__()
        self.encoder = encoder
        self.head    = ForecastHead(d_model, horizon)

    def forward(self, x):
        h_t = self.encoder(x)
        return self.head(h_t), h_t


# ─────────────────────────────────────────
# 6. Loss
# ─────────────────────────────────────────

def pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return torch.mean(torch.max(q * diff, (q - 1) * diff))


def compute_loss(y_true, mu):
    horizon = mu.shape[1]
    w = torch.ones(horizon, device=mu.device)
    w[:13] = 0.3
    w[13:] = 1.0

    loss = 0
    for k in range(horizon):
        l_k = (
            pinball_loss(y_true[:, k], mu[:, k], q=0.5) +
            pinball_loss(y_true[:, k], mu[:, k], q=0.7)
        )
        nonzero = (y_true[:, k] > 0)
        if nonzero.sum() > 0:
            l_k = l_k + 0.1 * F.mse_loss(
                mu[:, k][nonzero],
                y_true[:, k][nonzero]
            )
        loss += w[k] * l_k

    return loss / horizon


# ─────────────────────────────────────────
# 7. 训练
# ─────────────────────────────────────────

def train_model(model, train_loader, val_loader,
                n_epochs=30, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5)

    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            x = batch['x']
            y = batch['y']
            mu, h_t = model(x)
            loss = compute_loss(y, mu)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x']
                y = batch['y']
                mu, _ = model(x)
                val_loss += compute_loss(y, mu).item()

        scheduler.step()
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

    return train_losses, val_losses


# ─────────────────────────────────────────
# 8. Probing测试
# ─────────────────────────────────────────

def probing_test(model, val_loader):
    model.eval()
    all_h      = []
    all_regime = []

    with torch.no_grad():
        for batch in val_loader:
            x      = batch['x']
            regime = batch['regime']
            _, h_t = model(x)
            all_h.append(h_t.numpy())
            all_regime.append(regime.numpy())

    all_h      = np.concatenate(all_h,      axis=0)
    all_regime = np.concatenate(all_regime, axis=0)

    # 检查是否两个类别都有
    if len(np.unique(all_regime)) < 2:
        return 0.5  # 只有一个类别，无法分类

    n     = len(all_h)
    split = int(n * 0.8)
    X_tr, X_te = all_h[:split],      all_h[split:]
    y_tr, y_te = all_regime[:split], all_regime[split:]

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return 0.5

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    return clf.score(X_te, y_te)


# ─────────────────────────────────────────
# 9. Regime区分测试
# ─────────────────────────────────────────

def make_regime_input(demand_pattern, T=52):
    demand = np.array(demand_pattern, dtype=np.float32)
    v_t    = np.log1p(demand)
    b_t    = (demand > 0).astype(np.float32)

    d_t  = np.zeros(T, dtype=np.float32)
    last = -1
    for i in range(T):
        if b_t[i] > 0:
            last = i
        d_t[i] = (i - last) / 52.0 if last >= 0 else 1.0

    t        = np.arange(T)
    sin_year = np.sin(2 * np.pi * t / 52).astype(np.float32)
    cos_year = np.cos(2 * np.pi * t / 52).astype(np.float32)

    features = np.stack(
        [v_t, b_t, d_t, sin_year, cos_year], axis=1)
    return torch.tensor(features).unsqueeze(0)


def test_regime_discrimination(encoder):
    T = 52
    patterns = {
        'A_完全休眠': [0] * T,
        'B_规律激活': [3 if i % 4 == 0 else 0 for i in range(T)],
        'C_高频激活': [2] * T,
        'D_刚激活':   [0] * 44 + [2, 0, 3, 0, 1, 0, 4, 0],
        'E_季节性':   [4 if 20 <= i <= 35 else 0 for i in range(T)]
    }

    encoder.eval()
    h_vectors = {}
    with torch.no_grad():
        for name, demand in patterns.items():
            x = make_regime_input(demand, T)
            h = encoder(x)
            h_vectors[name] = h.squeeze(0)

    names = list(h_vectors.keys())
    n     = len(names)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = torch.norm(
                h_vectors[names[i]] - h_vectors[names[j]]
            ).item()
            dist_matrix[i, j] = dist

    upper = dist_matrix[np.triu_indices(n, k=1)]
    score = upper.mean()

    idx_sleep  = names.index('A_完全休眠')
    idx_active = names.index('C_高频激活')
    idx_new    = names.index('D_刚激活')

    return {
        'score':             score,
        'sleep_active_dist': dist_matrix[idx_sleep, idx_active],
        'sleep_new_dist':    dist_matrix[idx_sleep, idx_new],
        'h_vectors':         h_vectors,
        'names':             names
    }


# ─────────────────────────────────────────
# 10. 核心：不同零值率的对比实验
# ─────────────────────────────────────────

def run_experiment(zero_rate, encoder_classes):
    """
    给定零值率，跑完整的训练+评估
    返回每个Encoder的结果
    """
    print(f"\n{'─'*50}")
    print(f"零值率：{zero_rate:.0%}")

    # 生成数据
    df   = generate_data(n_asins=30, n_weeks=104,
                         target_zero_rate=zero_rate)
    data = make_features(df)

    actual_zero_rate = (df['Demand'] == 0).mean()
    print(f"实际零值率：{actual_zero_rate:.1%}")

    asins      = list(data.keys())
    train_data = {k: data[k] for k in asins[:24]}
    val_data   = {k: data[k] for k in asins[24:]}

    train_dataset = DemandDataset(train_data, history=52, horizon=20)
    val_dataset   = DemandDataset(val_data,   history=52, horizon=20)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(
        val_dataset,   batch_size=32, shuffle=False)

    results = {}

    for enc_class in encoder_classes:
        encoder = enc_class(input_dim=5, d_model=64)
        name    = encoder.name()

        model = Forecaster(encoder, d_model=64, horizon=20)

        # 训练
        train_model(model, train_loader, val_loader,
                   n_epochs=30, lr=1e-3)

        # Probing测试
        probing_acc = probing_test(model, val_loader)

        # Regime区分测试
        reg_result = test_regime_discrimination(model.encoder)

        results[name] = {
            'probing':           probing_acc,
            'regime_score':      reg_result['score'],
            'sleep_active_dist': reg_result['sleep_active_dist'],
            'sleep_new_dist':    reg_result['sleep_new_dist'],
            'actual_zero_rate':  actual_zero_rate
        }

        print(f"  {name:15s} | "
              f"Probing: {probing_acc:.3f} | "
              f"Regime分数: {reg_result['score']:.3f} | "
              f"刚激活距离: {reg_result['sleep_new_dist']:.3f}")

    return results, actual_zero_rate


# ─────────────────────────────────────────
# 11. 可视化
# ─────────────────────────────────────────

def plot_zero_rate_comparison(all_results, zero_rates):
    """
    画出不同零值率下三个Encoder的性能对比
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('不同零值率对Encoder学习Regime的影响',
                 fontsize=14)

    colors    = {'LSTM': 'steelblue', 'TCN': 'darkorange',
                 'Transformer': 'green', 'TCN+SparseAttn': 'purple'}
    enc_names = ['LSTM', 'TCN', 'Transformer', 'TCN+SparseAttn']

    actual_rates = [all_results[zr]['actual_zero_rate']
                    for zr in zero_rates]
    x = np.arange(len(zero_rates))
    width = 0.25

    metrics = [
        ('probing',      'Probing准确率\n（越高越好）',      0.5),
        ('regime_score', 'Regime区分分数\n（越高越好）',     0),
        ('sleep_new_dist','休眠vs刚激活距离\n（越高越好）',  0),
    ]

    for ax, (metric, title, baseline) in zip(axes, metrics):
        for i, enc_name in enumerate(enc_names):
            vals = [all_results[zr][enc_name][metric]
                    for zr in zero_rates]
            bars = ax.bar(x + i * width, vals, width,
                          label=enc_name,
                          color=colors[enc_name],
                          alpha=0.8)

            # 数值标注
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f'{v:.2f}',
                        ha='center', va='bottom', fontsize=7)

        if baseline > 0:
            ax.axhline(baseline, color='red', linestyle='--',
                       linewidth=1.5, alpha=0.7,
                       label=f'基线={baseline}')

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x + width)
        ax.set_xticklabels(
            [f'{r:.0%}' for r in actual_rates],
            fontsize=9
        )
        ax.set_xlabel('实际零值率')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
    plt.close()

    # 第二张图：折线图，看趋势
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle('零值率增加时各Encoder性能变化趋势',
                  fontsize=14)

    for ax, (metric, title, baseline) in zip(axes2, metrics):
        for enc_name in enc_names:
            vals = [all_results[zr][enc_name][metric]
                    for zr in zero_rates]
            ax.plot(actual_rates, vals,
                    'o-', label=enc_name,
                    color=colors[enc_name],
                    linewidth=2, markersize=8)

        if baseline > 0:
            ax.axhline(baseline, color='red', linestyle='--',
                       linewidth=1.5, alpha=0.7,
                       label=f'随机基线={baseline}')

        ax.set_title(title, fontsize=11)
        ax.set_xlabel('实际零值率')
        ax.set_xticks(actual_rates)
        ax.set_xticklabels([f'{r:.0%}' for r in actual_rates])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()


# ─────────────────────────────────────────
# 12. Main
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("="*55)
    print("零值率对Encoder学习Regime能力的影响")
    print("测试：25%, 50%, 75% 零值率")
    print("="*55)

    zero_rates    = [0.25, 0.50, 0.75]
    encoder_classes = [LSTMEncoder, TCNEncoder,
                       TransformerEncoder, TCNSparseAttnEncoder]

    all_results = {}

    for zr in zero_rates:
        results, actual = run_experiment(zr, encoder_classes)
        all_results[zr] = {}
        all_results[zr]['actual_zero_rate'] = actual
        for enc_name, metrics in results.items():
            all_results[zr][enc_name] = metrics

    # 汇总表格
    print("\n" + "="*70)
    print("汇总：不同零值率下的Probing准确率")
    print("="*70)
    print(f"{'零值率':>10} {'LSTM':>12} {'TCN':>12} "
          f"{'Transformer':>14} {'TCN+SparseAttn':>16}")
    print("-"*50)

    for zr in zero_rates:
        actual = all_results[zr]['actual_zero_rate']
        row    = f"{actual:>10.1%}"
        for enc in ['LSTM', 'TCN', 'Transformer', 'TCN+SparseAttn']:
            row += f"  {all_results[zr][enc]['probing']:>10.3f}"
        print(row)

    print("\n汇总：不同零值率下的Regime区分分数")
    print("-"*50)
    for zr in zero_rates:
        actual = all_results[zr]['actual_zero_rate']
        row    = f"{actual:>10.1%}"
        for enc in ['LSTM', 'TCN', 'Transformer', 'TCN+SparseAttn']:
            row += f"  {all_results[zr][enc]['regime_score']:>10.3f}"
        print(row)

    print("\n汇总：不同零值率下的休眠vs刚激活距离")
    print("-"*50)
    for zr in zero_rates:
        actual = all_results[zr]['actual_zero_rate']
        row    = f"{actual:>10.1%}"
        for enc in ['LSTM', 'TCN', 'Transformer', 'TCN+SparseAttn']:
            row += f"  {all_results[zr][enc]['sleep_new_dist']:>10.3f}"
        print(row)

    # 关键结论
    print("\n" + "="*55)
    print("关键问题：零值率越高，Encoder越难学到Regime吗？")
    print("="*55)

    for enc in ['LSTM', 'TCN', 'Transformer', 'TCN+SparseAttn']:
        probings = [all_results[zr][enc]['probing']
                    for zr in zero_rates]
        trend = "下降" if probings[-1] < probings[0] else "上升或持平"
        print(f"\n{enc}：")
        print(f"  25%→50%→75% Probing: "
              f"{probings[0]:.3f} → {probings[1]:.3f} → {probings[2]:.3f}")
        print(f"  趋势：{trend}")
        if probings[-1] < probings[0] - 0.1:
            print(f"  → 零值率增加显著影响了{enc}的Regime学习能力")
        elif probings[-1] > probings[0] - 0.05:
            print(f"  → {enc}对零值率变化较为鲁棒")

    # 可视化
    print("\n[生成图表...]")
    plot_zero_rate_comparison(all_results, zero_rates)

    print("\n完成！")