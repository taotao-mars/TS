"""
零值率对TCN+SparseAttn学习Regime能力的影响
测试：25%, 50%, 75% 零值率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────
# 1. 数据生成
# ─────────────────────────────────────────

def generate_data(n_asins=30, n_weeks=104, target_zero_rate=0.50):
    rows = []
    base_rate_scale = 1 - target_zero_rate
    for asin_id in range(n_asins):
        season_strength = np.random.uniform(0.2, 0.8)
        base_rate       = np.random.uniform(
            base_rate_scale * 0.5, base_rate_scale * 1.0)
        peak_week = np.random.randint(0, 52)
        for week in range(n_weeks):
            season = np.sin(2 * np.pi * (week - peak_week) / 52)
            p_buy  = np.clip(base_rate + season_strength * max(0, season),
                             0.01, 0.95)
            if np.random.rand() < p_buy:
                mu     = 3 + 5 * max(0, season)
                demand = min(np.random.negative_binomial(1, 1/(1+mu)), 50)
            else:
                demand = 0
            rows.append({'ASIN': f'ASIN_{asin_id:03d}',
                         'Week': pd.Timestamp('2023-10-01') +
                                 pd.Timedelta(weeks=week),
                         'Demand': int(demand)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 2. 特征工程
# ─────────────────────────────────────────

def make_features(df):
    df = df.copy()
    df['Week'] = pd.to_datetime(df['Week'])
    df = df.sort_values(['ASIN', 'Week']).reset_index(drop=True)
    df['t'] = ((df['Week'] - df['Week'].min()).dt.days // 7).astype(int)
    data = {}
    for asin, group in df.groupby('ASIN'):
        group  = group.reset_index(drop=True)
        demand = group['Demand'].values.astype(float)
        t      = group['t'].values
        T      = len(demand)
        v_t    = np.log1p(demand)
        b_t    = (demand > 0).astype(float)
        d_t    = np.zeros(T)
        last   = -1
        for i in range(T):
            if b_t[i] > 0: last = i
            d_t[i] = (i - last) / 52.0 if last >= 0 else 1.0
        features = np.stack([v_t, b_t, d_t,
                             np.sin(2*np.pi*t/52),
                             np.cos(2*np.pi*t/52)], axis=1).astype(np.float32)
        data[asin] = {'features': features, 'demand': demand.astype(np.float32)}
    return data


# ─────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────

class DemandDataset(Dataset):
    def __init__(self, data, history=52, horizon=20):
        self.samples = []
        for asin, d in data.items():
            features  = d['features']
            demand    = d['demand']
            T         = len(demand)
            hist_mean = max(demand[:history].mean()
                            if history <= T else demand.mean(), 0.01)
            for start in range(T - history - horizon + 1):
                x      = features[start:start+history]
                y      = demand[start+history:start+history+horizon]
                recent = demand[start+history-4:start+history].mean()
                prev   = demand[start+history-8:start+history-4].mean()
                # 四分类：0休眠 1刚激活 2稳定激活 3过渡期
                if recent < hist_mean * 0.2:
                    r4 = 0
                elif recent >= hist_mean * 0.8 and prev < hist_mean * 0.2:
                    r4 = 1
                elif recent >= hist_mean * 0.8:
                    r4 = 2
                else:
                    r4 = 3
                self.samples.append({
                    'x': torch.tensor(x), 'y': torch.tensor(y),
                    'regime2': 0 if r4 == 0 else 1, 'regime4': r4
                })

    def __len__(self):       return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# ─────────────────────────────────────────
# 4. TCN + Sparse Attention Encoder
# ─────────────────────────────────────────

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv    = nn.Conv1d(in_ch, out_ch,
                                 kernel_size=kernel_size, dilation=dilation)
    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class SparseAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, b_t):
        mask = (b_t == 0) & ~(b_t == 0).all(dim=1, keepdim=True)
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return self.norm(x + out)


class TCNSparseAttnEncoder(nn.Module):
    """TCN学0 + SparseAttn学非0 → h_t信息最完整"""
    def __init__(self, input_dim=5, d_model=64):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        dilations        = [1, 2, 3, 4, 8, 26, 52]
        self.convs       = nn.ModuleList([
            CausalConv1d(d_model, d_model, 2, d) for d in dilations])
        self.norms       = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in dilations])
        self.sparse_attn = SparseAttention(d_model)
        self.final_norm  = nn.LayerNorm(d_model)

    def forward(self, x):
        b_t = x[:, :, 1]
        h   = self.input_proj(x).permute(0, 2, 1)
        for conv, norm in zip(self.convs, self.norms):
            h = F.gelu(norm((conv(h) + h).permute(0, 2, 1)).permute(0, 2, 1))
        h = self.sparse_attn(h.permute(0, 2, 1), b_t)
        return self.final_norm(h[:, -1, :])

    def name(self): return "TCN+SparseAttn"


# ─────────────────────────────────────────
# 5. 训练
# ─────────────────────────────────────────

class Forecaster(nn.Module):
    def __init__(self, encoder, d_model=64, horizon=20):
        super().__init__()
        self.encoder = encoder
        self.head    = nn.Sequential(nn.Linear(d_model, d_model),
                                     nn.ReLU(),
                                     nn.Linear(d_model, horizon))
    def forward(self, x):
        h = self.encoder(x)
        return F.softplus(self.head(h)), h


def pinball_loss(y, p, q):
    d = y - p
    return torch.mean(torch.max(q * d, (q-1) * d))


def compute_loss(y, mu):
    w = torch.ones(mu.shape[1], device=mu.device)
    w[:13] = 0.3
    loss = 0
    for k in range(mu.shape[1]):
        l = pinball_loss(y[:,k], mu[:,k], 0.5) + \
            pinball_loss(y[:,k], mu[:,k], 0.7)
        nz = (y[:,k] > 0)
        if nz.sum() > 0:
            l = l + 0.1 * F.mse_loss(mu[:,k][nz], y[:,k][nz])
        loss += w[k] * l
    return loss / mu.shape[1]


def train_model(model, train_loader, val_loader, n_epochs=30):
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch  = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    for _ in range(n_epochs):
        model.train()
        for b in train_loader:
            mu, _ = model(b['x'])
            loss  = compute_loss(b['y'], mu)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        sch.step()


# ─────────────────────────────────────────
# 6. Probing测试
# ─────────────────────────────────────────

def probing_test(model, val_loader):
    model.eval()
    all_h, all_r2, all_r4 = [], [], []
    with torch.no_grad():
        for b in val_loader:
            _, h = model(b['x'])
            all_h.append(h.numpy())
            all_r2.append(b['regime2'].numpy())
            all_r4.append(b['regime4'].numpy())

    H  = np.concatenate(all_h)
    r2 = np.concatenate(all_r2)
    r4 = np.concatenate(all_r4)
    n  = len(H); sp = int(n * 0.8)

    sc = StandardScaler()
    Xt = sc.fit_transform(H[:sp])
    Xe = sc.transform(H[sp:])

    res = {}

    # Linear2
    res['linear2'] = LogisticRegression(max_iter=1000).fit(
        Xt, r2[:sp]).score(Xe, r2[sp:]) \
        if len(np.unique(r2[:sp])) >= 2 else 0.5

    # Linear4 + 刚激活Acc
    if len(np.unique(r4[:sp])) >= 2:
        clf4 = LogisticRegression(max_iter=1000).fit(Xt, r4[:sp])
        res['linear4'] = clf4.score(Xe, r4[sp:])
        pred = clf4.predict(Xe)
        m1   = (r4[sp:] == 1)
        res['new_active'] = (pred[m1] == 1).mean() if m1.sum() > 0 else 0.0
    else:
        res['linear4'] = 0.25; res['new_active'] = 0.0

    # MLP
    try:
        res['mlp'] = MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=500,
            random_state=42).fit(Xt, r2[:sp]).score(Xe, r2[sp:])
    except Exception:
        res['mlp'] = 0.5

    return res


# ─────────────────────────────────────────
# 7. Regime距离测试
# ─────────────────────────────────────────

def make_input(pattern, T=52):
    d    = np.array(pattern, dtype=np.float32)
    b    = (d > 0).astype(np.float32)
    dt   = np.zeros(T, dtype=np.float32)
    last = -1
    for i in range(T):
        if b[i] > 0: last = i
        dt[i] = (i - last) / 52.0 if last >= 0 else 1.0
    t = np.arange(T)
    f = np.stack([np.log1p(d), b, dt,
                  np.sin(2*np.pi*t/52).astype(np.float32),
                  np.cos(2*np.pi*t/52).astype(np.float32)], axis=1)
    return torch.tensor(f).unsqueeze(0)


def regime_dist_test(encoder):
    T = 52
    patterns = {
        'A_休眠':   [0] * T,
        'B_规律':   [3 if i%4==0 else 0 for i in range(T)],
        'C_高频':   [2] * T,
        'D_刚激活': [0]*44 + [2,0,3,0,1,0,4,0],
        'E_季节':   [4 if 20<=i<=35 else 0 for i in range(T)],
        'F_衰退':   [max(0, int(6-i/8)) for i in range(T)],
        'G_复苏':   [max(0, int(i/8-3)) for i in range(T)],
        'H_间歇':   [3 if i%8==0 else 0 for i in range(T)]
    }
    encoder.eval()
    hvecs = {}
    with torch.no_grad():
        for name, pat in patterns.items():
            hvecs[name] = encoder(make_input(pat, T)).squeeze(0)

    names = list(hvecs.keys())
    n     = len(names)
    D     = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i,j] = torch.norm(hvecs[names[i]] - hvecs[names[j]]).item()

    upper = D[np.triu_indices(n, k=1)]
    si    = names.index('A_休眠')
    di    = names.index('D_刚激活')
    return {'score': upper.mean(), 'sleep_new_dist': D[si, di]}


# ─────────────────────────────────────────
# 8. 主实验
# ─────────────────────────────────────────

def run_experiment(zero_rate):
    print(f"\n{'─'*50}")
    print(f"零值率：{zero_rate:.0%}")
    df     = generate_data(n_asins=30, n_weeks=104,
                           target_zero_rate=zero_rate)
    data   = make_features(df)
    actual = (df['Demand'] == 0).mean()
    print(f"实际零值率：{actual:.1%}")

    asins = list(data.keys())
    tl = DataLoader(DemandDataset({k: data[k] for k in asins[:24]}),
                    batch_size=32, shuffle=True)
    vl = DataLoader(DemandDataset({k: data[k] for k in asins[24:]}),
                    batch_size=32, shuffle=False)

    enc   = TCNSparseAttnEncoder(input_dim=5, d_model=64)
    model = Forecaster(enc, d_model=64, horizon=20)
    train_model(model, tl, vl, n_epochs=30)

    prob = probing_test(model, vl)
    reg  = regime_dist_test(model.encoder)

    print(f"  Linear2:    {prob['linear2']:.3f}  "
          f"Linear4: {prob['linear4']:.3f}  "
          f"MLP: {prob['mlp']:.3f}")
    print(f"  刚激活Acc:  {prob['new_active']:.3f}  "
          f"刚激活距离: {reg['sleep_new_dist']:.3f}  "
          f"总分: {reg['score']:.3f}")

    return {**prob, **reg, 'actual_zero_rate': actual}


# ─────────────────────────────────────────
# 9. 可视化
# ─────────────────────────────────────────

def plot_results(all_results, zero_rates):
    actual = [all_results[zr]['actual_zero_rate'] for zr in zero_rates]
    metrics = [
        ('linear2',       'Linear2 Probing\n(baseline=0.5)', 0.5),
        ('linear4',       'Linear4 Probing\n(baseline=0.25)', 0.25),
        ('sleep_new_dist','Sleep vs New-Active\nDistance', 0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('TCN+SparseAttn: Zero Rate vs Regime Learning', fontsize=13)
    for ax, (metric, title, bl) in zip(axes, metrics):
        vals = [all_results[zr][metric] for zr in zero_rates]
        ax.plot(actual, vals, 'o-', color='purple',
                linewidth=2.5, markersize=10)
        for x, v in zip(actual, vals):
            ax.annotate(f'{v:.3f}', (x, v),
                        textcoords='offset points',
                        xytext=(0, 10), ha='center', fontsize=10)
        if bl > 0:
            ax.axhline(bl, color='red', linestyle='--',
                       linewidth=1.5, label=f'Baseline={bl}')
            ax.legend(fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Zero Rate')
        ax.set_xticks(actual)
        ax.set_xticklabels([f'{r:.0%}' for r in actual])
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()


# ─────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("TCN+SparseAttn: Zero Rate Impact on Regime Learning")
    print("="*55)

    zero_rates  = [0.25, 0.50, 0.75]
    all_results = {}
    for zr in zero_rates:
        all_results[zr] = run_experiment(zr)

    print("\n" + "="*55)
    print("Summary")
    print("="*55)
    metrics = [
        ('linear2',       'Linear2 Probing'),
        ('linear4',       'Linear4 Probing'),
        ('mlp',           'MLP Probing'),
        ('new_active',    'New-Active Acc'),
        ('sleep_new_dist','Sleep-NewActive Dist'),
        ('score',         'Regime Score'),
    ]
    rates = [f"{all_results[zr]['actual_zero_rate']:.0%}"
             for zr in zero_rates]
    print(f"{'Metric':<22} {rates[0]:>10} {rates[1]:>10} {rates[2]:>10}")
    print("-"*55)
    for key, label in metrics:
        vals = [all_results[zr][key] for zr in zero_rates]
        best = max(vals)
        row  = f"{label:<22}"
        for v in vals:
            mark = "*" if abs(v-best) < 1e-6 else " "
            row += f"  {v:>7.3f}{mark}"
        print(row)
    print("* = best")

    best_zr = max(all_results,
                  key=lambda z: all_results[z]['sleep_new_dist'])
    print(f"\n-> Best zero rate for regime detection: "
          f"{all_results[best_zr]['actual_zero_rate']:.0%}")

    plot_results(all_results, zero_rates)
    print("\nDone!")
