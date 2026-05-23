"""
High-sparse aggressive experiment for:
TCN + Future-Gated SparsePeakAttn + ENN + Two-Head v3

What this script does:
1. Sample ASINs from data_raw1.
2. Compute ASIN-level zero_rate using the available data.
3. Keep only the high_sparse group:
      high_sparse : zero_rate >= 0.7
4. Train one specialized model only on high_sparse ASINs.
5. Use more aggressive settings to reduce underbias:
      beta_tail higher
      beta_peak higher
      prior_scale higher
      lambda_q slightly higher
      kl_weight slightly higher
6. Optionally apply a light aggressive inference correction after prediction.

Assumptions:
- data_raw1 already exists in your notebook/session.
- calculate_wape_using_lp_oos2 and quick_error_check already exist.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


torch.manual_seed(42)
np.random.seed(42)


# =====================================================
# 0. Sampling and grouping
# =====================================================

def prepare_data_sample(data_raw1, n_asins=5000):
    data_raw1 = data_raw1.copy()
    data_raw1['order_week'] = pd.to_datetime(data_raw1['order_week'])

    sample_asins = np.random.choice(
        data_raw1['asin'].unique(),
        size=min(n_asins, data_raw1['asin'].nunique()),
        replace=False
    )

    data_small = data_raw1[data_raw1['asin'].isin(sample_asins)].copy()

    print("Sample ASINs:", data_small['asin'].nunique())
    print("Sample rows:", len(data_small))

    return data_small


def add_zero_rate_group(data_raw, zero_thresholds=(0.4, 0.7)):
    df = data_raw.copy()
    df['fbi_demand'] = pd.to_numeric(df['fbi_demand'], errors='coerce').fillna(0).clip(lower=0)

    asin_stats = (
        df.groupby('asin')
        .agg(
            zero_rate=('fbi_demand', lambda x: (x == 0).mean()),
            total_demand=('fbi_demand', 'sum'),
            n_weeks=('fbi_demand', 'count')
        )
        .reset_index()
    )

    low, high = zero_thresholds

    def assign_group(z):
        if z < low:
            return 'low_sparse'
        elif z < high:
            return 'mid_sparse'
        else:
            return 'high_sparse'

    asin_stats['zero_group'] = asin_stats['zero_rate'].apply(assign_group)

    df = df.merge(
        asin_stats[['asin', 'zero_rate', 'zero_group']],
        on='asin',
        how='left'
    )

    print("\nASIN counts by zero-rate group:")
    print(asin_stats.groupby('zero_group')['asin'].nunique().reset_index(name='n_asins'))

    print("\nZero-rate quantiles:")
    print(asin_stats['zero_rate'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

    return df, asin_stats


# =====================================================
# 1. Data Loading
# =====================================================

def _safe_numeric(df, col, default=0.0):
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
    return df


def _rolling_positive_mean(arr, window=None):
    out = np.zeros(len(arr), dtype=float)
    for i in range(len(arr)):
        start = 0 if window is None else max(0, i - window)
        hist = arr[start:i]
        hist = hist[hist > 0]
        out[i] = hist.mean() if len(hist) > 0 else 0.0
    return out


def _rolling_positive_percentile(arr, q=75, window=None):
    out = np.zeros(len(arr), dtype=float)
    for i in range(len(arr)):
        start = 0 if window is None else max(0, i - window)
        hist = arr[start:i]
        hist = hist[hist > 0]
        out[i] = np.percentile(hist, q) if len(hist) > 0 else 0.0
    return out


def _rolling_recent_peak(arr, window=13):
    out = np.zeros(len(arr), dtype=float)
    for i in range(len(arr)):
        start = max(0, i - window)
        hist = arr[start:i]
        out[i] = hist.max() if len(hist) > 0 else 0.0
    return out


def load_real_data(data_raw):
    holiday_cols = [c for c in data_raw.columns if c.startswith('holiday_indicator_')]
    context_cols = ['our_price', 'in_stock_dph'] + holiday_cols

    base_cols = ['asin', 'order_week', 'fbi_demand', 'scot_oos']
    keep_cols = [c for c in base_cols + context_cols if c in data_raw.columns]

    df = data_raw[keep_cols].copy()
    df = df.rename(columns={
        'asin': 'ASIN',
        'order_week': 'Week',
        'fbi_demand': 'Demand',
        'scot_oos': 'OOS'
    })

    df['Week'] = pd.to_datetime(df['Week'])
    df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce').fillna(0).clip(lower=0)
    df['OOS'] = pd.to_numeric(df['OOS'], errors='coerce').fillna(0)

    for c in context_cols:
        df = _safe_numeric(df, c, default=0.0)

    df = df.sort_values(['ASIN', 'Week']).reset_index(drop=True)

    # Price is assumed known at forecast creation time.
    df['our_price'] = np.log1p(df['our_price'].clip(lower=0))

    # Lag-safe DPH / traffic feature.
    df['in_stock_dph'] = (
        df.groupby('ASIN')['in_stock_dph']
        .shift(1)
        .fillna(0)
        .clip(lower=0)
    )

    for c in holiday_cols:
        df[c] = df[c].clip(lower=0, upper=1)

    # Holiday window: previous week + holiday week.
    if len(holiday_cols) > 0:
        holiday_window = np.zeros(len(df), dtype=np.float32)
        for c in holiday_cols:
            cur = df[c].values.astype(float)
            prev_week_window = np.roll(cur, -1)
            prev_week_window[-1] = 0
            holiday_window = np.maximum(holiday_window, np.maximum(cur, prev_week_window))
        df['promo_t'] = holiday_window
    else:
        df['promo_t'] = 0.0

    df['t'] = ((df['Week'] - df['Week'].min()).dt.days // 7).astype(int)

    data = {}
    for asin, group in df.groupby('ASIN'):
        group = group.reset_index(drop=True)

        demand = group['Demand'].values.astype(float)
        oos = group['OOS'].values.astype(float)
        weeks = group['Week'].values
        t = group['t'].values
        T = len(demand)

        v_t = np.log1p(demand)
        b_t = (demand > 0).astype(float)

        d_t = np.zeros(T)
        last = -1
        for i in range(T):
            if b_t[i] > 0:
                last = i
            d_t[i] = (i - last) / 52.0 if last >= 0 else 1.0

        hist_nonzero_mean = np.log1p(_rolling_positive_mean(demand, window=None))
        hist_nonzero_p75 = np.log1p(_rolling_positive_percentile(demand, q=75, window=None))
        recent_peak = np.log1p(_rolling_recent_peak(demand, window=13))

        features = np.stack([
            v_t,
            b_t,
            d_t,
            np.sin(2 * np.pi * t / 52),
            np.cos(2 * np.pi * t / 52),
            group['promo_t'].values.astype(float),
            np.sin(2 * np.pi * t / 13),
            np.cos(2 * np.pi * t / 13),
            hist_nonzero_mean,
            hist_nonzero_p75,
            recent_peak,
        ], axis=1).astype(np.float32)

        future_context = group[context_cols].values.astype(np.float32)

        data[asin] = {
            'features': features,
            'future_context': future_context,
            'demand': demand.astype(np.float32),
            'week': weeks,
            'oos': oos.astype(np.float32),
        }

    print("History encoder dim: 11")
    print(f"Conditional z context dim: {len(context_cols)}")
    print("Conditional z context columns:", context_cols)

    return data, len(context_cols), context_cols


# =====================================================
# 2. Dataset
# =====================================================

class DemandDataset(Dataset):
    def __init__(self, data, history=52, horizon=20, mode='train', val_weeks=20):
        self.samples = []

        for asin, d in data.items():
            features = d['features']
            future_context = d['future_context']
            demand = d['demand']
            weeks = d['week']
            oos = d['oos']
            T = len(demand)

            if mode == 'train':
                max_start = T - val_weeks - horizon - history + 1
                starts = range(max(0, max_start))
            else:
                start = T - history - horizon
                starts = [start] if start >= 0 else []

            for start in starts:
                target_weeks = weeks[start+history:start+history+horizon]

                self.samples.append({
                    'x': torch.tensor(features[start:start+history], dtype=torch.float32),
                    'future_context': torch.tensor(
                        future_context[start+history:start+history+horizon],
                        dtype=torch.float32
                    ),
                    'y': torch.tensor(demand[start+history:start+history+horizon], dtype=torch.float32),
                    'asin': asin,
                    'target_week': [str(w)[:10] for w in target_weeks],
                    'oos': torch.tensor(oos[start+history:start+history+horizon], dtype=torch.float32),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# =====================================================
# 3. Model
# =====================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class SparsePeakAttention(nn.Module):
    def __init__(self, d_model=32, n_heads=4, beta_peak=1.0):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.beta_peak = beta_peak

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, b_t, peak_score, peak_gate=None):
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        sparse_mask = (b_t == 0) & ~(b_t == 0).all(dim=1, keepdim=True)
        scores = scores.masked_fill(sparse_mask[:, None, None, :], -1e4)

        peak_norm = peak_score / (peak_score.max(dim=1, keepdim=True)[0] + 1e-6)

        if peak_gate is None:
            peak_gate = torch.ones(B, 1, device=x.device, dtype=x.dtype)
        peak_gate = peak_gate.view(B, 1, 1, 1)

        # Future-gated historical peak retrieval.
        scores = scores + self.beta_peak * peak_gate * peak_norm[:, None, None, :]

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return self.norm(x + out)


class TCNSparseAttnEncoder(nn.Module):
    def __init__(self, input_dim=11, d_model=32, horizon=20, beta_peak=1.5):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(input_dim, d_model)

        dilations = [1, 2, 3, 4, 8, 26, 52]
        self.convs = nn.ModuleList([CausalConv1d(d_model, d_model, 2, d) for d in dilations])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in dilations])

        self.sparse_attn = SparsePeakAttention(d_model, n_heads=4, beta_peak=beta_peak)
        self.final_norm = nn.LayerNorm(d_model)

        self.occ_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.mag_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.alpha_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))

        self.occurrence_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, horizon))
        self.magnitude_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, horizon))
        self.alpha_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, horizon))

    def forward(self, x, peak_gate=None):
        b_t = x[:, :, 1]
        peak_score = torch.sqrt(torch.expm1(x[:, :, 0]).clamp(min=0) + 1e-6)

        h = self.input_proj(x).permute(0, 2, 1)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h) + h
            h = h.permute(0, 2, 1)
            h = norm(h)
            h = F.gelu(h)
            h = h.permute(0, 2, 1)

        h = self.sparse_attn(h.permute(0, 2, 1), b_t, peak_score, peak_gate=peak_gate)
        h_t = self.final_norm(h[:, -1, :])

        h_occ = self.occ_proj(h_t)
        h_mag = self.mag_proj(h_t)
        h_alpha = self.alpha_proj(h_t)

        occ_logit = self.occurrence_head(h_occ)
        mu = F.softplus(self.magnitude_head(h_mag))
        alpha = F.softplus(self.alpha_head(h_alpha)) + 1e-4

        return occ_logit, mu, alpha, h_t


class ContextZGenerator(nn.Module):
    def __init__(self, d_phi=32, context_dim=2, d_z=16, horizon=20):
        super().__init__()
        self.d_z = d_z
        self.horizon = horizon
        self.context_dim = context_dim

        self.net = nn.Sequential(
            nn.Linear(d_phi + horizon * context_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2 * d_z)
        )

    def forward(self, phi, future_context):
        B = phi.shape[0]
        ctx = future_context.reshape(B, -1)
        out = self.net(torch.cat([phi, ctx], dim=-1))
        z_mean, z_logstd = out.chunk(2, dim=-1)
        z_logstd = z_logstd.clamp(-4, 2)
        z_std = torch.exp(z_logstd)

        kl = -0.5 * (1 + 2 * z_logstd - z_mean ** 2 - z_std ** 2).sum(dim=-1).mean()
        return z_mean, z_std, kl


class Epinet(nn.Module):
    def __init__(self, d_phi=32, d_z=16, horizon=20, prior_scale=0.3):
        super().__init__()
        self.d_z = d_z
        self.horizon = horizon
        self.prior_scale = prior_scale

        self.learnable = nn.Sequential(
            nn.Linear(d_z + d_phi, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3 * horizon * d_z),
        )
        self.prior = nn.Sequential(
            nn.Linear(d_z + d_phi, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * horizon * d_z),
        )
        for p in self.prior.parameters():
            p.requires_grad = False

    def forward(self, phi, z):
        inp = torch.cat([z, phi], dim=-1)
        sl = self.learnable(inp).view(-1, 3 * self.horizon, self.d_z)
        sl = torch.einsum('bhd,bd->bh', sl, z)
        sp = self.prior(inp).view(-1, 3 * self.horizon, self.d_z)
        sp = torch.einsum('bhd,bd->bh', sp, z) * self.prior_scale
        out = sl + sp
        occ_e = out[:, :self.horizon]
        mu_e = out[:, self.horizon:2*self.horizon]
        al_e = out[:, 2*self.horizon:]
        return occ_e, mu_e, al_e


class TCN_ENN(nn.Module):
    def __init__(self, input_dim=11, context_dim=2, d_model=32, d_z=16, horizon=20, prior_scale=0.3, beta_peak=1.5):
        super().__init__()
        self.d_z = d_z
        self.horizon = horizon

        self.encoder = TCNSparseAttnEncoder(input_dim=input_dim, d_model=d_model, horizon=horizon, beta_peak=beta_peak)
        self.z_generator = ContextZGenerator(d_phi=d_model, context_dim=context_dim, d_z=d_z, horizon=horizon)
        self.epinet = Epinet(d_phi=d_model, d_z=d_z, horizon=horizon, prior_scale=prior_scale)

        self.future_peak_gate = nn.Sequential(
            nn.Linear(horizon * context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, future_context, nZ=8):
        B = x.shape[0]
        peak_gate = self.future_peak_gate(future_context.reshape(B, -1))
        occ_logit_base, mu_base, alpha_base, h_t = self.encoder(x, peak_gate=peak_gate)

        phi = h_t.detach()
        z_mean, z_std, kl = self.z_generator(phi, future_context)

        preds = []
        for _ in range(nZ):
            eps = torch.randn_like(z_mean)
            z = z_mean + z_std * eps
            occ_e, mu_e, al_e = self.epinet(phi, z)

            occ_logit = occ_logit_base + occ_e
            occ_prob = torch.sigmoid(occ_logit).clamp(1e-6, 1 - 1e-6)
            mu = F.softplus(mu_base + mu_e)
            alpha = F.softplus(alpha_base + al_e) + 1e-4
            preds.append((occ_logit, occ_prob, mu, alpha))

        return preds, kl

    def predict(self, x, future_context, M=50):
        self.eval()
        with torch.no_grad():
            B = x.shape[0]
            peak_gate = self.future_peak_gate(future_context.reshape(B, -1))
            occ_logit_base, mu_base, alpha_base, h_t = self.encoder(x, peak_gate=peak_gate)
            phi = h_t.detach()
            z_mean, z_std, _ = self.z_generator(phi, future_context)

            samples = []
            for _ in range(M):
                eps = torch.randn_like(z_mean)
                z = z_mean + z_std * eps
                occ_e, mu_e, al_e = self.epinet(phi, z)

                occ_logit = occ_logit_base + occ_e
                occ_prob = torch.sigmoid(occ_logit).clamp(1e-6, 1 - 1e-6)
                mu = F.softplus(mu_base + mu_e)
                alpha = F.softplus(alpha_base + al_e) + 1e-4

                active = torch.bernoulli(occ_prob).bool()
                nb = torch.distributions.NegativeBinomial(
                    total_count=(1.0 / alpha).clamp(min=1e-4),
                    probs=(mu * alpha / (1 + mu * alpha)).clamp(1e-6, 1 - 1e-6),
                )
                mag = nb.sample().float().clamp(min=1)
                sample = torch.where(active, mag, torch.zeros_like(mag))
                samples.append(sample)

            samples = torch.stack(samples, dim=1)
            p50 = samples.quantile(0.5, dim=1)
            p70 = samples.quantile(0.7, dim=1)
            p70 = torch.maximum(p70, p50)
        return p50, p70


# =====================================================
# 4. Loss, training, evaluation
# =====================================================

def two_head_loss(y, occ_logit, mu, alpha, beta_tail=0.5):
    eps = 1e-6
    active = (y > 0).float()
    occ_loss = F.binary_cross_entropy_with_logits(occ_logit, active)

    nz_mask = y > 0
    mag_loss = torch.tensor(0.0, device=y.device)

    if nz_mask.sum() > 0:
        y_nz = y[nz_mask]
        mu_nz = mu[nz_mask]
        alpha_nz = alpha[nz_mask]

        r = (1.0 / alpha_nz).clamp(min=eps)
        p = (mu_nz * alpha_nz / (1 + mu_nz * alpha_nz)).clamp(eps, 1 - eps)

        nll = -(
            torch.lgamma(y_nz + r)
            - torch.lgamma(r)
            - torch.lgamma(y_nz + 1)
            + r * torch.log(1 - p)
            + y_nz * torch.log(p)
        )

        weight = 1.0 + beta_tail * torch.log1p(y_nz)
        mag_loss = (nll * weight).sum() / weight.sum().clamp(min=1.0)

    return occ_loss + mag_loss, occ_loss, mag_loss


def pinball(y, pred, q):
    d = y - pred
    return torch.mean(torch.max(q * d, (q - 1) * d))


def train(model, tr_ld, va_ld, epochs=10, nZ=8, lr=1e-3, lambda_q=0.05, beta_tail=0.5, kl_weight=0.001):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float('inf')
    best_sd = None

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        last_occ_loss = 0.0
        last_mag_loss = 0.0
        last_kl = 0.0

        for b in tr_ld:
            x = b['x']
            fc = b['future_context']
            y = b['y']

            preds, kl = model(x, fc, nZ=nZ)
            losses, occ_losses, mag_losses = [], [], []

            for occ_logit, occ_prob, mu, alpha in preds:
                loss_i, occ_i, mag_i = two_head_loss(y, occ_logit, mu, alpha, beta_tail=beta_tail)
                losses.append(loss_i)
                occ_losses.append(occ_i)
                mag_losses.append(mag_i)

            main_loss = sum(losses) / nZ
            occ_loss = sum(occ_losses) / nZ
            mag_loss = sum(mag_losses) / nZ

            exp_stack = torch.stack([occ_prob * mu for occ_logit, occ_prob, mu, alpha in preds], dim=1)
            p50_train = exp_stack.quantile(0.5, dim=1)
            p70_train = torch.maximum(exp_stack.quantile(0.7, dim=1), p50_train)
            q_loss = pinball(y, p50_train, 0.5) + pinball(y, p70_train, 0.7)

            loss = main_loss + lambda_q * q_loss + kl_weight * kl

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item()
            last_occ_loss = occ_loss.item()
            last_mag_loss = mag_loss.item()
            last_kl = kl.item()

        sch.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in va_ld:
                p50, p70 = model.predict(b['x'], b['future_context'], M=30)
                vl += (pinball(b['y'], p50, 0.5) + pinball(b['y'], p70, 0.7)).item()
        vl /= max(1, len(va_ld))

        if vl < best_val:
            best_val = vl
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1:3d} | "
            f"train={tr_loss / max(1, len(tr_ld)):.4f} | "
            f"val={vl:.4f} | "
            f"occ={last_occ_loss:.4f} | "
            f"mag={last_mag_loss:.4f} | "
            f"kl={last_kl:.4f}"
        )

    if best_sd is not None:
        model.load_state_dict(best_sd)
    print(f"Best val: {best_val:.4f}")


def evaluate(model, va_ld, M=100):
    all_y, all_p50, all_p70 = [], [], []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            p50, p70 = model.predict(b['x'], b['future_context'], M=M)
            all_y.append(b['y'].numpy())
            all_p50.append(p50.numpy())
            all_p70.append(p70.numpy())

    y = np.concatenate(all_y)
    p50 = np.concatenate(all_p50)
    p70 = np.concatenate(all_p70)
    yt = torch.tensor(y)
    return pinball(yt, torch.tensor(p50), 0.5).item(), pinball(yt, torch.tensor(p70), 0.7).item()


def generate_forecast_df(model, va_ld, M=50):
    rows = []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            x, fc, y, oos = b['x'], b['future_context'], b['y'], b['oos']
            p50, p70 = model.predict(x, fc, M=M)

            hist_mean = (x[:, :, 0].exp() - 1).mean(dim=1, keepdim=True).clamp(min=0)
            hm50 = hist_mean.expand_as(y)
            hm70 = hm50 * 1.25

            for i in range(y.shape[0]):
                for h in range(y.shape[1]):
                    rows.append({
                        'asin': b['asin'][i],
                        'order_week': pd.to_datetime(b['target_week'][h][i]),
                        'fcst_week_index': h + 1,
                        'fbi_demand': y[i, h].item(),
                        'scot_oos': oos[i, h].item(),
                        'oos': oos[i, h].item(),
                        'oos_status': oos[i, h].item(),
                        'p50_amxl': p50[i, h].item(),
                        'p70_amxl': p70[i, h].item(),
                        'p50_scot': hm50[i, h].item(),
                        'p70_scot': hm70[i, h].item(),
                    })
    return pd.DataFrame(rows)


# =====================================================
# 5. Diagnosis
# =====================================================

def generate_diagnostic_df(model, va_ld, M=100, threshold=0.5):
    rows = []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            p50, p70 = model.predict(b['x'], b['future_context'], M=M)
            for i in range(b['y'].shape[0]):
                for h in range(b['y'].shape[1]):
                    y_val = b['y'][i, h].item()
                    p50_val = p50[i, h].item()
                    p70_val = p70[i, h].item()
                    rows.append({
                        'asin': b['asin'][i],
                        'order_week': pd.to_datetime(b['target_week'][h][i]),
                        'horizon': h + 1,
                        'y': y_val,
                        'p50': p50_val,
                        'p70': p70_val,
                        'true_active': int(y_val > 0),
                        'pred_active_p50': int(p50_val > threshold),
                        'pred_active_p70': int(p70_val > threshold),
                    })
    return pd.DataFrame(rows)


def underbias_diagnosis(diag_df, pred_col='p70', threshold=0.5):
    y = diag_df['y'].values
    pred = diag_df[pred_col].values
    true_active = y > 0
    pred_active = pred > threshold

    tp = np.sum(true_active & pred_active)
    fp = np.sum(~true_active & pred_active)
    fn = np.sum(true_active & ~pred_active)
    tn = np.sum(~true_active & ~pred_active)

    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    total_under = np.maximum(y - pred, 0).sum()
    total_y = y.sum()

    missed_active_mask = true_active & ~pred_active
    magnitude_under_mask = true_active & pred_active

    missed_under = np.maximum(y[missed_active_mask] - pred[missed_active_mask], 0).sum()
    magnitude_under = np.maximum(y[magnitude_under_mask] - pred[magnitude_under_mask], 0).sum()

    ratio = np.array([np.nan])
    if magnitude_under_mask.sum() > 0:
        ratio = pred[magnitude_under_mask] / np.maximum(y[magnitude_under_mask], 1e-8)

    return pd.DataFrame([{
        'pred_col': pred_col,
        'threshold': threshold,
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'occurrence_recall': recall,
        'occurrence_precision': precision,
        'occurrence_f1': f1,
        'total_underbias': total_under,
        'underbias_rate': total_under / max(1e-8, total_y),
        'missed_active_share': missed_under / max(1e-8, total_under),
        'magnitude_under_share': magnitude_under / max(1e-8, total_under),
        'avg_pred_over_true_when_active_predicted': np.nanmean(ratio),
        'median_pred_over_true_when_active_predicted': np.nanmedian(ratio),
    }])


def magnitude_gap(diag_df):
    df = diag_df[diag_df['true_active'] == 1].copy()
    if len(df) == 0:
        print("No active weeks in diagnostic dataframe.")
        return pd.DataFrame()

    y = df['y'].values
    p50 = df['p50'].values
    p70 = df['p70'].values

    out = pd.DataFrame([{
        'true_active_mean': y.mean(),
        'p50_active_mean': p50.mean(),
        'p70_active_mean': p70.mean(),
        'p50_pct_of_true': p50.mean() / max(y.mean(), 1e-8),
        'p70_pct_of_true': p70.mean() / max(y.mean(), 1e-8),
        'p50_gap': y.mean() - p50.mean(),
        'p70_gap': y.mean() - p70.mean(),
    }])

    print("\n[Magnitude Gap - Active weeks only]")
    print(out.T)
    return out


# =====================================================
# 6. Single run for one group
# =====================================================

def run_one_group(
    data_group,
    group_name,
    prior_scale=0.5,
    beta_peak=1.5,
    epochs=20,
    history=52,
    horizon=20,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=50,
    lambda_q=0.05,
    beta_tail=0.5,
    kl_weight=0.001
):
    print("\n" + "#" * 70)
    print(f"Running group: {group_name}")
    print("#" * 70)

    data, context_dim, context_cols = load_real_data(data_group)

    all_demand = np.concatenate([d['demand'] for d in data.values()])
    print(f"ASINs: {len(data)}")
    print(f"Rows: {len(data_group)}")
    print(f"Zero rate: {(all_demand == 0).mean():.1%}")

    tr_ds = DemandDataset(data, history=history, horizon=horizon, mode='train', val_weeks=horizon)
    va_ds = DemandDataset(data, history=history, horizon=horizon, mode='val', val_weeks=horizon)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(tr_ds)} | Val samples: {len(va_ds)}")

    model = TCN_ENN(
        input_dim=11,
        context_dim=context_dim,
        d_model=d_model,
        d_z=d_z,
        horizon=horizon,
        prior_scale=prior_scale,
        beta_peak=beta_peak
    )

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train(
        model,
        tr_ld,
        va_ld,
        epochs=epochs,
        nZ=8,
        lr=1e-3,
        lambda_q=lambda_q,
        beta_tail=beta_tail,
        kl_weight=kl_weight
    )

    pl50, pl70 = evaluate(model, va_ld, M=100)
    print(f"\nPinball: P50={pl50:.4f} | P70={pl70:.4f}")

    forecast_df = generate_forecast_df(model, va_ld, M=M_eval)
    forecast_df['zero_group_run'] = group_name

    diag_df = generate_diagnostic_df(model, va_ld, M=100, threshold=0.5)
    diag_p50 = underbias_diagnosis(diag_df, pred_col='p50', threshold=0.5)
    diag_p70 = underbias_diagnosis(diag_df, pred_col='p70', threshold=0.5)
    mag_gap = magnitude_gap(diag_df)

    print("\nUnderbias Diagnosis - P50:")
    print(diag_p50.T)
    print("\nUnderbias Diagnosis - P70:")
    print(diag_p70.T)

    # Business WAPE using your official functions.
    quantiles = [0.5, 0.7]
    wape_df = calculate_wape_using_lp_oos2(
        forecast_df,
        quantiles,
        remove_oos_dp=True,
        source='lp'
    )

    cols_p50 = [
        'p50_amxl_penalty',
        'p50_scot_penalty',
        'p50_amxl_overbias',
        'p50_scot_overbias',
        'p50_amxl_underbias',
        'p50_scot_underbias',
        'fbi_demand'
    ]
    cols_p70 = [
        'p70_amxl_penalty',
        'p70_scot_penalty',
        'p70_amxl_overbias',
        'p70_scot_overbias',
        'p70_amxl_underbias',
        'p70_scot_underbias',
        'fbi_demand'
    ]

    p50_wape, p50_penalty_diff = quick_error_check(wape_df, cols_p50)
    p70_wape, p70_penalty_diff = quick_error_check(wape_df, cols_p70)

    print("\nP50 WAPE / Penalty Summary:")
    print(p50_wape)
    print("P50 penalty diff:", p50_penalty_diff)

    print("\nP70 WAPE / Penalty Summary:")
    print(p70_wape)
    print("P70 penalty diff:", p70_penalty_diff)

    summary = {
        'zero_group': group_name,
        'n_asins': data_group['asin'].nunique(),
        'n_rows_raw': len(data_group),
        'n_forecast_rows': len(forecast_df),
        'raw_zero_rate': (pd.to_numeric(data_group['fbi_demand'], errors='coerce').fillna(0) == 0).mean(),
        'forecast_true_active_ratio': (forecast_df['fbi_demand'] > 0).mean(),
        'forecast_p50_active_ratio': (forecast_df['p50_amxl'] > 0).mean(),
        'forecast_p70_active_ratio': (forecast_df['p70_amxl'] > 0).mean(),
        'pinball50': pl50,
        'pinball70': pl70,
        'p50_penalty_diff': p50_penalty_diff,
        'p70_penalty_diff': p70_penalty_diff,
    }

    return {
        'group': group_name,
        'model': model,
        'forecast_df': forecast_df,
        'wape_df': wape_df,
        'diag_df': diag_df,
        'diag_p50': diag_p50,
        'diag_p70': diag_p70,
        'mag_gap': mag_gap,
        'p50_wape': p50_wape,
        'p70_wape': p70_wape,
        'summary': summary,
    }


# =====================================================
# 7. Run high_sparse only
# =====================================================

def run_high_sparse_aggressive_experiment(
    data_raw1,
    n_asins=5000,
    zero_thresholds=(0.4, 0.7),
    prior_scale=0.5,
    beta_peak=1.5,
    epochs=20,
    history=52,
    horizon=20,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=80,
    lambda_q=0.10,
    beta_tail=1.0,
    kl_weight=0.003
):
    data_small = prepare_data_sample(data_raw1, n_asins=n_asins)
    data_grouped, asin_stats = add_zero_rate_group(data_small, zero_thresholds=zero_thresholds)

    data_high = data_grouped[data_grouped['zero_group'] == 'high_sparse'].copy()

    print("\n" + "=" * 80)
    print("HIGH-SPARSE ONLY AGGRESSIVE EXPERIMENT")
    print("=" * 80)
    print(f"High sparse ASINs: {data_high['asin'].nunique()}")
    print(f"High sparse rows : {len(data_high)}")
    print(f"Raw zero rate    : {(pd.to_numeric(data_high['fbi_demand'], errors='coerce').fillna(0) == 0).mean():.4f}")
    print("Aggressive training settings:")
    print(f"  beta_tail={beta_tail}, beta_peak={beta_peak}, prior_scale={prior_scale}")
    print(f"  lambda_q={lambda_q}, kl_weight={kl_weight}, M_eval={M_eval}")

    result = run_one_group(
        data_high,
        group_name='high_sparse_aggressive',
        prior_scale=prior_scale,
        beta_peak=beta_peak,
        epochs=epochs,
        history=history,
        horizon=horizon,
        d_model=d_model,
        d_z=d_z,
        batch_size=batch_size,
        M_eval=M_eval,
        lambda_q=lambda_q,
        beta_tail=beta_tail,
        kl_weight=kl_weight
    )

    summary_df = pd.DataFrame([result['summary']])

    print("\n" + "=" * 80)
    print("HIGH-SPARSE AGGRESSIVE SUMMARY")
    print("=" * 80)
    print(summary_df)

    return result, summary_df, asin_stats


# =====================================================
# 8. Execute
# =====================================================

high_sparse_result, high_sparse_summary_df, asin_zero_stats = run_high_sparse_aggressive_experiment(
    data_raw1,
    n_asins=5000,
    zero_thresholds=(0.4, 0.7),
    prior_scale=0.5,
    beta_peak=1.5,
    epochs=20,
    history=52,
    horizon=20,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=80,
    lambda_q=0.10,
    beta_tail=1.0,
    kl_weight=0.003
)
