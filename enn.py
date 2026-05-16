"""
Demand Forecasting: TCN+SparseAttn Encoder + ENN
======================================================
Architecture (Osband et al. 2021 - Epistemic Neural Networks):
  f(x, z) = base_net(x) + epinet(sg[h_t], z)
  epinet   = ProjectedMLP: mlp([z, phi])^T · z  (Eq.6)
  prior    = same architecture, fixed weights (no grad)

Training (Algorithm 1):
  Sample nZ epistemic indices z ~ N(0,I) per batch
  Average NegBin NLL over nZ samples
  Encoder trained via epinet output; phi uses stop_gradient

Output:
  NegativeBinomial(mu, alpha) -> P50/P70 over 20-week horizon
  Joint prediction: z fixed across all 20 weeks

Usage:
  # In SageMaker / Jupyter:
  data = load_real_data(data_raw1)
  # Then run main()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════
# 1. Data Loading
# ══════════════════════════════════════════════════════

def load_real_data(data_raw):
    """
    Load demand data from DataFrame.

    Required columns:
      asin        : product identifier
      order_week  : week start date (e.g. '2023-10-01')
      fbi_demand  : weekly demand (float, NaN -> 0)
      scot_oos    : out-of-stock flag (0/1)

    Returns:
      dict: {asin -> {'features': np.array [T,5], 'demand': np.array [T]}}

    Features (5-dim per timestep):
      [0] log(1 + demand)          : log-scaled demand
      [1] I(demand > 0)            : purchase indicator (sparse attention mask)
      [2] weeks_since_last_sale/52 : time since last purchase
      [3] sin(2*pi*t/52)           : seasonal encoding
      [4] cos(2*pi*t/52)           : seasonal encoding
    """
    df = data_raw[['asin', 'order_week', 'fbi_demand', 'scot_oos']].copy()
    df.columns = ['ASIN', 'Week', 'Demand', 'OOS']
    df['Week']   = pd.to_datetime(df['Week'])
    df['Demand'] = df['Demand'].fillna(0).clip(lower=0)
    df['OOS']    = df['OOS'].fillna(0)
    df = df.sort_values(['ASIN', 'Week']).reset_index(drop=True)
    df['t'] = ((df['Week'] - df['Week'].min()).dt.days // 7).astype(int)

    data = {}
    for asin, group in df.groupby('ASIN'):
        group  = group.reset_index(drop=True)
        demand = group['Demand'].values.astype(float)
        t      = group['t'].values
        T      = len(demand)

        v_t  = np.log1p(demand)
        b_t  = (demand > 0).astype(float)
        d_t  = np.zeros(T)
        last = -1
        for i in range(T):
            if b_t[i] > 0: last = i
            d_t[i] = (i - last) / 52.0 if last >= 0 else 1.0

        features = np.stack([
            v_t,
            b_t,
            d_t,
            np.sin(2 * np.pi * t / 52),
            np.cos(2 * np.pi * t / 52),
        ], axis=1).astype(np.float32)

        data[asin] = {
            'features': features,
            'demand':   demand.astype(np.float32),
        }
    return data


# ══════════════════════════════════════════════════════
# 2. Dataset  (time-based train/val split)
# ══════════════════════════════════════════════════════

class DemandDataset(Dataset):
    """
    Rolling-window dataset with time-based train/val split.

    Train: rolling windows where prediction target does NOT overlap last val_weeks.
    Val:   each ASIN's final prediction point (last horizon weeks).

    This prevents information leakage: val targets are strictly after train targets.

    Args:
      data      : output of load_real_data()
      history   : input sequence length (weeks)
      horizon   : forecast horizon (weeks) -- paper uses 20
      mode      : 'train' or 'val'
      val_weeks : number of weeks reserved for validation
    """
    def __init__(self, data, history=52, horizon=20, mode='train', val_weeks=20):
        self.samples = []
        for asin, d in data.items():
            features = d['features']
            demand   = d['demand']
            T        = len(demand)

            if mode == 'train':
                max_start = T - val_weeks - horizon - history + 1
                for start in range(max(0, max_start)):
                    self.samples.append({
                        'x': torch.tensor(features[start:start+history]),
                        'y': torch.tensor(demand[start+history:start+history+horizon]),
                    })
            else:  # val
                start = T - history - horizon
                if start >= 0:
                    self.samples.append({
                        'x': torch.tensor(features[start:start+history]),
                        'y': torch.tensor(demand[start+history:start+history+horizon]),
                    })

    def __len__(self):        return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# ══════════════════════════════════════════════════════
# 3. Encoder  (TCN + SparseAttention)
# ══════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv    = nn.Conv1d(in_ch, out_ch,
                                 kernel_size=kernel_size, dilation=dilation)
    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class SparseAttention(nn.Module):
    """Attend only to purchase (non-zero) positions."""
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
    """
    TCN dilations=[1,2,3,4,8,26,52]: captures multi-scale zero patterns.
    SparseAttention: captures relationships between purchase events.
    h_t [B, d_model]: full regime representation.

    Two output heads for NegativeBinomial parameters:
      mu    [B, horizon]: mean
      alpha [B, horizon]: dispersion  (variance = mu + alpha * mu^2)
    """
    def __init__(self, input_dim=5, d_model=64, horizon=20):
        super().__init__()
        self.horizon     = horizon
        self.input_proj  = nn.Linear(input_dim, d_model)
        dilations        = [1, 2, 3, 4, 8, 26, 52]
        self.convs       = nn.ModuleList([
            CausalConv1d(d_model, d_model, 2, d) for d in dilations])
        self.norms       = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in dilations])
        self.sparse_attn = SparseAttention(d_model, n_heads=4)
        self.final_norm  = nn.LayerNorm(d_model)
        self.base_head   = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, horizon))
        self.alpha_head  = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, horizon))

    def forward(self, x):
        b_t = x[:, :, 1]
        h   = self.input_proj(x).permute(0, 2, 1)
        for conv, norm in zip(self.convs, self.norms):
            h = F.gelu(norm((conv(h) + h).permute(0, 2, 1)).permute(0, 2, 1))
        h     = self.sparse_attn(h.permute(0, 2, 1), b_t)
        h_t   = self.final_norm(h[:, -1, :])
        mu    = F.softplus(self.base_head(h_t))
        alpha = F.softplus(self.alpha_head(h_t)) + 1e-4
        return mu, alpha, h_t


# ══════════════════════════════════════════════════════
# 4. Epinet  (ProjectedMLP, Osband et al. Eq.6)
# ══════════════════════════════════════════════════════

class Epinet(nn.Module):
    """
    Learnable:  mlp([z, phi])  reshaped to [B, 2H, d_z], dot z -> [B, 2H]
    Prior:      same architecture, random init, requires_grad=False
    Output:     (mu_correction [B,H], alpha_correction [B,H])

    Concat order: z first, then phi  (matches official ProjectedMLP)
    """
    def __init__(self, d_phi=64, d_z=16, horizon=20, prior_scale=0.0):
        super().__init__()
        self.d_z         = d_z
        self.horizon     = horizon
        self.prior_scale = prior_scale

        self.learnable = nn.Sequential(
            nn.Linear(d_z + d_phi, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2 * horizon * d_z),
        )
        self.prior = nn.Sequential(
            nn.Linear(d_z + d_phi, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * horizon * d_z),
        )
        for p in self.prior.parameters():
            p.requires_grad = False

    def forward(self, phi, z):
        inp = torch.cat([z, phi], dim=-1)
        sl  = self.learnable(inp).view(-1, 2*self.horizon, self.d_z)
        sl  = torch.einsum('bhd,bd->bh', sl, z)
        sp  = self.prior(inp).view(-1, 2*self.horizon, self.d_z)
        sp  = torch.einsum('bhd,bd->bh', sp, z) * self.prior_scale
        out = sl + sp
        return out[:, :self.horizon], out[:, self.horizon:]


# ══════════════════════════════════════════════════════
# 5. Full Model  (TCN_ENN)
# ══════════════════════════════════════════════════════

class TCN_ENN(nn.Module):
    """
    Fixed-z ENN:
      z ~ N(0, I)  per sample, fixed across 20-week horizon -> Joint prediction.
      Encoder trains via NegBin NLL on epinet output.
      Epinet trained with stop_gradient on phi.
    """
    def __init__(self, input_dim=5, d_model=64, d_z=16,
                 horizon=20, prior_scale=0.0):
        super().__init__()
        self.d_z     = d_z
        self.horizon = horizon
        self.encoder = TCNSparseAttnEncoder(input_dim, d_model, horizon)
        self.epinet  = Epinet(d_phi=d_model, d_z=d_z,
                              horizon=horizon, prior_scale=prior_scale)

    def forward(self, x, nZ=8):
        """
        Returns list of nZ (mu, alpha) tuples.
        Each mu/alpha: [B, horizon]  NegBin parameters.
        z is sampled per-sample (not per-batch) to allow ASIN-level variation.
        """
        mu_base, alpha_base, h_t = self.encoder(x)
        phi = h_t.detach()   # stop_gradient: epinet does not backprop into encoder

        preds = []
        for _ in range(nZ):
            z         = torch.randn(x.shape[0], self.d_z, device=x.device)
            mu_e, al_e = self.epinet(phi, z)
            mu    = F.softplus(mu_base + mu_e)
            alpha = F.softplus(alpha_base + al_e) + 1e-4
            preds.append((mu, alpha))
        return preds

    def predict(self, x, M=200):
        """
        Sample M trajectories from NegBin, return P50 and P70.
        z fixed per trajectory -> joint prediction across 20 weeks.
        """
        self.eval()
        with torch.no_grad():
            mu_base, alpha_base, h_t = self.encoder(x)
            phi = h_t.detach()
            samples = []
            for _ in range(M):
                z      = torch.randn(x.shape[0], self.d_z, device=x.device)
                mu_e, al_e = self.epinet(phi, z)
                mu    = F.softplus(mu_base + mu_e)
                alpha = F.softplus(alpha_base + al_e) + 1e-4
                dist  = torch.distributions.NegativeBinomial(
                    total_count=(1.0 / alpha).clamp(min=1e-4),
                    probs=(mu * alpha / (1 + mu * alpha)).clamp(1e-6, 1 - 1e-6),
                )
                samples.append(dist.sample().float())
            samples = torch.stack(samples, dim=1)   # [B, M, H]
            p50 = samples.quantile(0.5, dim=1)
            p70 = samples.quantile(0.7, dim=1)
            p70 = torch.maximum(p70, p50)
        return p50, p70


# ══════════════════════════════════════════════════════
# 6. Loss & Metrics
# ══════════════════════════════════════════════════════

def negbin_nll(y, mu, alpha):
    """
    Negative Binomial negative log-likelihood.
    y:     [B, H]  observed demand (integer-valued)
    mu:    [B, H]  predicted mean
    alpha: [B, H]  dispersion  (var = mu + alpha * mu^2)
    """
    eps = 1e-6
    r   = (1.0 / alpha).clamp(min=eps)
    p   = (mu * alpha / (1 + mu * alpha)).clamp(eps, 1 - eps)
    nll = -(torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
            + r * torch.log(1 - p) + y * torch.log(p))
    return nll.mean()


def pinball(y, pred, q):
    d = y - pred
    return torch.mean(torch.max(q * d, (q - 1) * d))


# ══════════════════════════════════════════════════════
# 7. Training
# ══════════════════════════════════════════════════════

def train(model, tr_ld, va_ld, epochs=60, nZ=8, lr=1e-3):
    """
    Algorithm 1 (Osband et al.):
      For each batch, sample nZ epistemic indices z.
      Compute NegBin NLL for each z, average, backprop.
    """
    opt      = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = float('inf')
    best_sd  = None

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for b in tr_ld:
            x, y  = b['x'], b['y']
            preds = model(x, nZ=nZ)
            loss  = sum(negbin_nll(y, mu, alpha)
                        for mu, alpha in preds) / nZ
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        sch.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for b in va_ld:
                    p50, p70 = model.predict(b['x'], M=50)
                    vl += (pinball(b['y'], p50, 0.5) +
                           pinball(b['y'], p70, 0.7)).item()
            model.train()
            vl /= len(va_ld)
            if vl < best_val:
                best_val = vl
                best_sd  = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1:3d} | "
                  f"train={tr_loss/len(tr_ld):.4f} | val={vl:.4f}")

    if best_sd:
        model.load_state_dict(best_sd)


# ══════════════════════════════════════════════════════
# 8. Evaluation
# ══════════════════════════════════════════════════════

def evaluate(model, va_ld, M=200):
    """Evaluate P50/P70 Pinball Loss on validation set."""
    all_y, all_p50, all_p70 = [], [], []
    for b in va_ld:
        p50, p70 = model.predict(b['x'], M=M)
        all_y.append(b['y'].numpy())
        all_p50.append(p50.numpy())
        all_p70.append(p70.numpy())
    y   = np.concatenate(all_y)
    p50 = np.concatenate(all_p50)
    p70 = np.concatenate(all_p70)
    yt  = torch.tensor(y)
    pl50 = pinball(yt, torch.tensor(p50), 0.5).item()
    pl70 = pinball(yt, torch.tensor(p70), 0.7).item()
    return pl50, pl70


def hist_mean_baseline(va_ld):
    """HistMean baseline: predict history mean for all future weeks."""
    all_y, all_hm = [], []
    for b in va_ld:
        y         = b['y']
        hist_mean = (b['x'][:, :, 0].exp() - 1).mean(dim=1, keepdim=True).clamp(min=0)
        all_y.append(y)
        all_hm.append(hist_mean.expand_as(y))
    y_all  = torch.cat(all_y)
    hm_all = torch.cat(all_hm)
    hm50 = pinball(y_all, hm_all, 0.5).item()
    hm70 = pinball(y_all, hm_all * 1.25, 0.7).item()
    return hm50, hm70


# ══════════════════════════════════════════════════════
# 9. Main
# ══════════════════════════════════════════════════════

def main(data_raw1):
    """
    Entry point.

    Args:
      data_raw1 : DataFrame with columns [asin, order_week, fbi_demand, scot_oos]
    """
    print("TCN+SparseAttn + ENN  |  Sparse Demand Forecasting")
    print("=" * 55)

    # Load data
    data = load_real_data(data_raw1)
    all_demand = np.concatenate([d['demand'] for d in data.values()])
    print(f"ASINs: {len(data)}")
    print(f"Zero rate: {(all_demand == 0).mean():.1%}")

    # Time-based split
    tr_ld = DataLoader(
        DemandDataset(data, history=52, horizon=20, mode='train', val_weeks=20),
        batch_size=64, shuffle=True)
    va_ld = DataLoader(
        DemandDataset(data, history=52, horizon=20, mode='val', val_weeks=20),
        batch_size=64, shuffle=False)
    print(f"Train samples: {len(tr_ld.dataset)} | Val samples: {len(va_ld.dataset)}")

    # Model
    model = TCN_ENN(input_dim=5, d_model=64, d_z=16, horizon=20, prior_scale=0.0)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    # Train
    print("\nTraining (nZ=8 z-samples per batch, NegBin NLL)...")
    train(model, tr_ld, va_ld, epochs=60, nZ=8)

    # Evaluate
    pl50, pl70 = evaluate(model, va_ld, M=200)
    hm50, hm70 = hist_mean_baseline(va_ld)

    print(f"\n{'Method':<20} {'Pinball-50':>12} {'Pinball-70':>12}")
    print("-" * 46)
    print(f"{'HistMean':<20} {hm50:>12.4f} {hm70:>12.4f}")
    print(f"{'ENN (Fixed-z)':<20} {pl50:>12.4f} {pl70:>12.4f}")
    print(f"{'ENN vs HistMean':<20} {(hm50-pl50)/hm50*100:>+11.1f}% "
          f"{(hm70-pl70)/hm70*100:>+11.1f}%")

    return model


# ── Run ───────────────────────────────────────────────
# In SageMaker/Jupyter:
#   model = main(data_raw1)
#
# As script:
if __name__ == "__main__":
    raise RuntimeError(
        "Run this as a module in Jupyter/SageMaker:\n"
        "  from enn_amxl import main\n"
        "  model = main(data_raw1)"
    )
