"""
High-sparse experiment:
TCN + SparsePeakAttention + ENN + Anchored LogNormal Magnitude

变更 v2:
- d_model: 32 → 24  (减少过拟合)
- d_z: 16 → 8       (减少过拟合)
- batch_size: 64 → 128  (每个 batch 更多 active weeks)
- weight_decay: 1e-5 → 1e-4  (更强正则化)
- occ_pos_weight: 4.0 → 2.5  (更接近理论值，稳定 occ 梯度)
- 早停: patience=5  (防止过拟合继续)
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
    data_raw1["order_week"] = pd.to_datetime(data_raw1["order_week"])
    sample_asins = np.random.choice(
        data_raw1["asin"].unique(),
        size=min(n_asins, data_raw1["asin"].nunique()),
        replace=False
    )
    data_small = data_raw1[data_raw1["asin"].isin(sample_asins)].copy()
    print("Sample ASINs:", data_small["asin"].nunique())
    print("Sample rows:", len(data_small))
    return data_small


def add_zero_rate_group(data_raw, zero_thresholds=(0.4, 0.7)):
    df = data_raw.copy()
    df["fbi_demand"] = pd.to_numeric(
        df["fbi_demand"], errors="coerce"
    ).fillna(0).clip(lower=0)

    asin_stats = (
        df.groupby("asin")
        .agg(
            zero_rate=("fbi_demand", lambda x: (x == 0).mean()),
            total_demand=("fbi_demand", "sum"),
            n_weeks=("fbi_demand", "count"),
        )
        .reset_index()
    )

    low, high = zero_thresholds

    def assign_group(z):
        if z < low:
            return "low_sparse"
        elif z < high:
            return "mid_sparse"
        else:
            return "high_sparse"

    asin_stats["zero_group"] = asin_stats["zero_rate"].apply(assign_group)
    df = df.merge(
        asin_stats[["asin", "zero_rate", "zero_group"]],
        on="asin", how="left"
    )

    print("\nASIN counts by zero-rate group:")
    print(asin_stats.groupby("zero_group")["asin"].nunique().reset_index(name="n_asins"))
    print("\nZero-rate quantiles:")
    print(asin_stats["zero_rate"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

    return df, asin_stats


# =====================================================
# 1. Data loading
# =====================================================

def _safe_numeric(df, col, default=0.0):
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return df


def _rolling_positive_mean(arr, window=52):
    out = np.zeros(len(arr), dtype=float)
    for i in range(len(arr)):
        start = max(0, i - window)
        hist = arr[start:i]
        hist = hist[hist > 0]
        out[i] = hist.mean() if len(hist) > 0 else 0.0
    return out


def _rolling_positive_percentile(arr, q=75, window=52):
    out = np.zeros(len(arr), dtype=float)
    for i in range(len(arr)):
        start = max(0, i - window)
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
    holiday_cols = [c for c in data_raw.columns if c.startswith("holiday_indicator_")]
    context_cols = ["our_price", "in_stock_dph"] + holiday_cols

    static_cols = [
        "pkg_weight", "pkg_height", "pkg_length", "pkg_width",
        "gl_product_group", "category_code", "brand_class",
    ]
    static_cols_available = [c for c in static_cols if c in data_raw.columns]

    base_cols = ["asin", "order_week", "fbi_demand", "scot_oos"]
    keep_cols = [
        c for c in base_cols + context_cols + static_cols_available
        if c in data_raw.columns
    ]

    df = data_raw[keep_cols].copy()
    df = df.rename(columns={
        "asin": "ASIN", "order_week": "Week",
        "fbi_demand": "Demand", "scot_oos": "OOS",
    })

    df["Week"] = pd.to_datetime(df["Week"])
    df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce").fillna(0).clip(lower=0)
    df["OOS"] = pd.to_numeric(df["OOS"], errors="coerce").fillna(0)

    for c in context_cols:
        df = _safe_numeric(df, c, default=0.0)

    df = df.sort_values(["ASIN", "Week"]).reset_index(drop=True)
    df["our_price"] = np.log1p(df["our_price"].clip(lower=0))
    df["in_stock_dph"] = (
        df.groupby("ASIN")["in_stock_dph"].shift(1).fillna(0).clip(lower=0)
    )

    for c in holiday_cols:
        df[c] = df[c].clip(lower=0, upper=1)

    if len(holiday_cols) > 0:
        holiday_window = np.zeros(len(df), dtype=np.float32)
        for c in holiday_cols:
            cur = df[c].values.astype(float)
            prev_week = np.roll(cur, -1)
            prev_week[-1] = 0
            holiday_window = np.maximum(holiday_window, np.maximum(cur, prev_week))
        df["promo_t"] = holiday_window
    else:
        df["promo_t"] = 0.0

    df["t"] = ((df["Week"] - df["Week"].min()).dt.days // 7).astype(int)

    static_df = df.groupby("ASIN")[static_cols_available].first().reset_index()
    for c in static_cols_available:
        if static_df[c].dtype == object or str(static_df[c].dtype) == "category":
            static_df[c] = pd.Categorical(static_df[c]).codes.astype(float)
            static_df[c] = (static_df[c] - static_df[c].mean()) / (static_df[c].std() + 1e-8)
        else:
            static_df[c] = pd.to_numeric(static_df[c], errors="coerce").fillna(0).clip(lower=0)
            static_df[c] = np.log1p(static_df[c])
            static_df[c] = (static_df[c] - static_df[c].mean()) / (static_df[c].std() + 1e-8)

    static_lookup = static_df.set_index("ASIN")[static_cols_available].to_dict("index")
    n_static = len(static_cols_available)

    data = {}
    for asin, group in df.groupby("ASIN"):
        group = group.reset_index(drop=True)
        demand = group["Demand"].values.astype(float)
        oos = group["OOS"].values.astype(float)
        weeks = group["Week"].values
        t = group["t"].values

        v_t = np.log1p(demand)
        b_t = (demand > 0).astype(float)

        d_t = np.zeros(len(demand))
        last = -1
        for i in range(len(demand)):
            if b_t[i] > 0:
                last = i
            d_t[i] = (i - last) / 52.0 if last >= 0 else 1.0

        hist_nonzero_mean = np.log1p(_rolling_positive_mean(demand, window=52))
        hist_nonzero_p75  = np.log1p(_rolling_positive_percentile(demand, q=75, window=52))
        recent_peak       = np.log1p(_rolling_recent_peak(demand, window=13))

        features = np.stack([
            v_t, b_t, d_t,
            np.sin(2 * np.pi * t / 52),
            np.cos(2 * np.pi * t / 52),
            group["promo_t"].values.astype(float),
            np.sin(2 * np.pi * t / 13),
            np.cos(2 * np.pi * t / 13),
            hist_nonzero_mean,
            hist_nonzero_p75,
            recent_peak,
        ], axis=1).astype(np.float32)

        future_context = group[context_cols].values.astype(np.float32)

        if asin in static_lookup and n_static > 0:
            static_vec = np.array(
                [static_lookup[asin].get(c, 0.0) for c in static_cols_available],
                dtype=np.float32
            )
        else:
            static_vec = np.zeros(max(n_static, 1), dtype=np.float32)

        data[asin] = {
            "features": features,
            "future_context": future_context,
            "demand": demand.astype(np.float32),
            "week": weeks,
            "oos": oos.astype(np.float32),
            "static": static_vec,
        }

    print("History encoder dim: 11")
    print(f"Static feature dim: {n_static} {static_cols_available}")
    print(f"Conditional z context dim: {len(context_cols)}")

    return data, len(context_cols), context_cols, n_static


# =====================================================
# 2. Dataset
# =====================================================

class DemandDataset(Dataset):
    def __init__(self, data, history=52, horizon=20, mode="train", val_weeks=20):
        self.samples = []

        for asin, d in data.items():
            features = d["features"]
            future_context = d["future_context"]
            demand = d["demand"]
            weeks = d["week"]
            oos = d["oos"]
            T = len(demand)

            if mode == "train":
                max_start = T - val_weeks - horizon - history + 1
                starts = range(max(0, max_start))
            else:
                start = T - history - horizon
                starts = [start] if start >= 0 else []

            for start in starts:
                y_window = demand[start + history:start + history + horizon]
                target_weeks = weeks[start + history:start + history + horizon]

                self.samples.append({
                    "x": torch.tensor(features[start:start + history], dtype=torch.float32),
                    "future_context": torch.tensor(
                        future_context[start + history:start + history + horizon],
                        dtype=torch.float32
                    ),
                    "y": torch.tensor(y_window, dtype=torch.float32),
                    "asin": asin,
                    "target_week": [str(w)[:10] for w in target_weeks],
                    "oos": torch.tensor(
                        oos[start + history:start + history + horizon],
                        dtype=torch.float32
                    ),
                    "static": torch.tensor(d["static"], dtype=torch.float32),
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
    def __init__(self, d_model=24, n_heads=4, beta_peak=1.5):
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
        scores = scores + self.beta_peak * peak_gate * peak_norm[:, None, None, :]

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return self.norm(x + out)


class TCNSparseAttnEncoder(nn.Module):
    ANCHOR_IDXS = [8, 9, 10]  # hist_nonzero_mean, hist_nonzero_p75, recent_peak

    def __init__(self, input_dim=11, d_model=24, horizon=20,
                 beta_peak=1.5, static_dim=0):
        super().__init__()
        self.horizon = horizon
        self.static_dim = static_dim
        self.input_proj = nn.Linear(input_dim, d_model)

        dilations = [1, 2, 3, 4, 8, 26, 52]
        self.convs = nn.ModuleList([CausalConv1d(d_model, d_model, 2, d) for d in dilations])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in dilations])

        self.sparse_attn = SparsePeakAttention(d_model=d_model, n_heads=4, beta_peak=beta_peak)
        self.final_norm = nn.LayerNorm(d_model)

        d_static_emb = 16 if static_dim > 0 else 0
        n_anchors = len(self.ANCHOR_IDXS)

        self.occ_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model)
        )
        self.loc_proj = nn.Sequential(
            nn.Linear(d_model + n_anchors + d_static_emb, d_model),
            nn.ReLU(), nn.LayerNorm(d_model)
        )
        self.scale_proj = nn.Sequential(
            nn.Linear(d_model + n_anchors + d_static_emb, d_model),
            nn.ReLU(), nn.LayerNorm(d_model)
        )

        self.occurrence_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, horizon)
        )
        self.loc_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, horizon)
        )
        self.scale_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, horizon)
        )

        # 小随机初始化：训练开始时 loc ≈ anchor（tanh(小值)≈0）
        # 不能用全零初始化 weight，否则梯度消失，loc_head 永远不更新
        nn.init.normal_(self.loc_head[-1].weight, std=0.01)
        nn.init.zeros_(self.loc_head[-1].bias)

    def forward(self, x, peak_gate=None, static_emb=None):
        b_t = x[:, :, 1]
        peak_score = torch.sqrt(torch.expm1(x[:, :, 0]).clamp(min=0) + 1e-6)
        anchor_feats = x[:, -1, self.ANCHOR_IDXS]

        loc_anchor = (
            0.40 * anchor_feats[:, 0]
            + 0.40 * anchor_feats[:, 1]
            + 0.20 * anchor_feats[:, 2]
        ).unsqueeze(1).expand(-1, self.horizon)

        h = self.input_proj(x).permute(0, 2, 1)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h) + h
            h = h.permute(0, 2, 1)
            h = norm(h)
            h = F.gelu(h)
            h = h.permute(0, 2, 1)

        h = self.sparse_attn(h.permute(0, 2, 1), b_t, peak_score, peak_gate=peak_gate)
        h_t = self.final_norm(h[:, -1, :])

        # occurrence：完整梯度回流到 encoder
        # encoder 专注学习 occurrence 的时序表示
        h_occ = self.occ_proj(h_t)
        occ_logit_base = self.occurrence_head(h_occ)

        # magnitude：detach h_t，切断 LogNormal 梯度对 encoder 的干扰
        # z 通过 epinet 的 loc_e 仍然影响 magnitude，joint prediction 不受影响
        h_t_mag = h_t.detach()
        mag_inp = [h_t_mag, anchor_feats]
        if static_emb is not None:
            mag_inp.append(static_emb)
        mag_inp = torch.cat(mag_inp, dim=-1)

        h_loc = self.loc_proj(mag_inp)
        h_scale = self.scale_proj(mag_inp)

        loc_resid_raw = self.loc_head(h_loc)
        loc_base = loc_anchor + 0.30 * torch.tanh(loc_resid_raw)
        scale_raw = self.scale_head(h_scale)
        scale = 0.10 + 1.40 * torch.sigmoid(scale_raw)

        loc_base = torch.nan_to_num(loc_base, nan=0.0, posinf=6.0, neginf=-3.0).clamp(-3.0, 6.0)
        scale = torch.nan_to_num(scale, nan=1.0, posinf=1.5, neginf=0.10).clamp(0.10, 1.50)

        return occ_logit_base, loc_base, scale, h_t


class ContextZGenerator(nn.Module):
    def __init__(self, d_phi=24, context_dim=2, d_z=8, horizon=20):
        super().__init__()
        self.d_z = d_z
        self.horizon = horizon
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
    def __init__(self, d_phi=24, d_z=8, horizon=20, prior_scale=0.5):
        super().__init__()
        self.d_z = d_z
        self.horizon = horizon
        self.prior_scale = prior_scale

        self.learnable = nn.Sequential(
            nn.Linear(d_phi + d_z, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2 * horizon * d_z)
        )
        self.prior = nn.Sequential(
            nn.Linear(d_phi + d_z, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * horizon * d_z)
        )
        for p in self.prior.parameters():
            p.requires_grad = False

    def forward(self, phi, z):
        inp = torch.cat([phi, z], dim=-1)
        sl = self.learnable(inp).view(-1, 2 * self.horizon, self.d_z)
        sl = torch.einsum("bhd,bd->bh", sl, z)
        sp = self.prior(inp).view(-1, 2 * self.horizon, self.d_z)
        sp = torch.einsum("bhd,bd->bh", sp, z) * self.prior_scale
        out = sl + sp
        occ_e = out[:, :self.horizon]
        loc_e = out[:, self.horizon:]
        return occ_e, loc_e


class TCN_ENN_LogNormal(nn.Module):
    def __init__(
        self,
        input_dim=11,
        context_dim=2,
        d_model=24,
        d_z=8,
        horizon=20,
        prior_scale=0.5,
        beta_peak=1.5,
        static_dim=0,
        occ_enn_scale=0.15,
        loc_enn_scale=0.50,
    ):
        super().__init__()
        self.d_z = d_z
        self.horizon = horizon
        self.static_dim = static_dim
        self.occ_enn_scale = occ_enn_scale
        self.loc_enn_scale = loc_enn_scale

        self.encoder = TCNSparseAttnEncoder(
            input_dim=input_dim, d_model=d_model,
            horizon=horizon, beta_peak=beta_peak, static_dim=static_dim
        )

        if static_dim > 0:
            self.static_encoder = nn.Sequential(
                nn.Linear(static_dim, 32), nn.ReLU(),
                nn.LayerNorm(32), nn.Linear(32, 16), nn.ReLU()
            )
            d_cond = d_model + 16
        else:
            self.static_encoder = None
            d_cond = d_model

        self.z_generator = ContextZGenerator(
            d_phi=d_cond, context_dim=context_dim, d_z=d_z, horizon=horizon
        )
        self.epinet = Epinet(
            d_phi=d_cond, d_z=d_z, horizon=horizon, prior_scale=prior_scale
        )
        self.future_peak_gate = nn.Sequential(
            nn.Linear(horizon * context_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def _encode(self, x, future_context, static=None):
        B = x.shape[0]
        peak_gate = self.future_peak_gate(future_context.reshape(B, -1))

        if self.static_encoder is not None and static is not None:
            static_emb = self.static_encoder(static)
        else:
            static_emb = None

        occ_logit_base, loc_base, scale_base, h_t = self.encoder(
            x, peak_gate=peak_gate, static_emb=static_emb
        )

        if static_emb is not None:
            phi = torch.cat([h_t, static_emb], dim=-1)
        else:
            phi = h_t

        return occ_logit_base, loc_base, scale_base, phi

    def forward(self, x, future_context, nZ=8, static=None):
        occ_logit_base, loc_base, scale_base, phi = self._encode(x, future_context, static)
        phi_det = phi.detach()
        z_mean, z_std, kl = self.z_generator(phi_det, future_context)

        preds = []
        for _ in range(nZ):
            eps = torch.randn_like(z_mean)
            z = z_mean + z_std * eps
            occ_e, loc_e = self.epinet(phi_det, z)

            occ_logit = occ_logit_base + self.occ_enn_scale * torch.tanh(occ_e)
            occ_prob = torch.sigmoid(occ_logit).clamp(1e-6, 1 - 1e-6)
            loc = loc_base + self.loc_enn_scale * torch.tanh(loc_e)
            loc = torch.nan_to_num(loc, nan=0.0, posinf=6.0, neginf=-3.0).clamp(-3.0, 6.0)
            scale = scale_base

            preds.append((occ_logit, occ_prob, loc, scale))

        return preds, kl

    def predict(self, x, future_context, M=200, static=None):
        self.eval()
        with torch.no_grad():
            occ_logit_base, loc_base, scale_base, phi = self._encode(x, future_context, static)
            z_mean, z_std, _ = self.z_generator(phi, future_context)

            samples = []
            for _ in range(M):
                eps_z = torch.randn_like(z_mean)
                z = z_mean + z_std * eps_z
                occ_e, loc_e = self.epinet(phi, z)

                occ_logit = occ_logit_base + self.occ_enn_scale * torch.tanh(occ_e)
                occ_prob = torch.sigmoid(occ_logit).clamp(1e-6, 1 - 1e-6)
                loc = loc_base + self.loc_enn_scale * torch.tanh(loc_e)
                loc = torch.nan_to_num(loc, nan=0.0, posinf=6.0, neginf=-3.0).clamp(-3.0, 6.0)
                scale = scale_base.clamp(0.10, 1.50)

                active = torch.bernoulli(occ_prob).bool()
                eps_mag = torch.randn_like(loc)
                mag = torch.exp(loc + scale * eps_mag) - 1
                mag = mag.clamp(min=0)
                sample = torch.where(active, mag, torch.zeros_like(mag))
                samples.append(sample)

            samples = torch.stack(samples, dim=1)
            p50 = samples.quantile(0.5, dim=1)
            p70 = samples.quantile(0.7, dim=1)
            p70 = torch.maximum(p70, p50)

        return p50, p70


# =====================================================
# 4. Loss / Train / Eval
# =====================================================

def lognormal_enn_loss(y, occ_logit, loc, scale, occ_pos_weight=1.5):
    active = (y > 0).float()
    pos_weight = torch.tensor(occ_pos_weight, device=y.device, dtype=y.dtype)
    occ_loss = F.binary_cross_entropy_with_logits(occ_logit, active, pos_weight=pos_weight)

    pos_mask = y > 0
    if pos_mask.sum() == 0:
        mag_nll = torch.tensor(0.0, device=y.device)
        return occ_loss, occ_loss, mag_nll

    loc = torch.nan_to_num(loc, nan=0.0, posinf=6.0, neginf=-3.0).clamp(-3.0, 6.0)
    scale = torch.nan_to_num(scale, nan=1.0, posinf=1.5, neginf=0.10).clamp(0.10, 1.50)

    log_y = torch.log1p(y[pos_mask])
    dist = torch.distributions.Normal(loc[pos_mask], scale[pos_mask])
    mag_nll = -dist.log_prob(log_y).mean()

    return occ_loss + mag_nll, occ_loss, mag_nll


def pinball(y, pred, q):
    d = y - pred
    return torch.mean(torch.max(q * d, (q - 1) * d))


def train(
    model,
    tr_ld,
    va_ld,
    epochs=30,
    nZ=8,
    lr=5e-4,
    kl_weight=0.003,
    occ_pos_weight=1.5,    # 4.0 → 2.5：更接近理论值，稳定梯度
    M_val=100,
    patience=5,            # 早停
):
    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4   # 1e-5 → 1e-4：更强正则化
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    best_sd = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        last_occ = last_mag = last_kl = 0.0

        # ── 诊断（只在第一个 epoch 的前3个 batch 打印）──
        _diag_batches = 3 if epoch == 0 else 0

        for _bi, b in enumerate(tr_ld):
            x = b["x"]
            fc = b["future_context"]
            y = b["y"]
            static = b.get("static", None)

            if _bi < _diag_batches:
                active_cnt = (y > 0).sum().item()
                total_cnt  = y.numel()
                print(
                    f"  [diag batch {_bi}] "
                    f"active={active_cnt}/{total_cnt} "
                    f"({100*active_cnt/total_cnt:.1f}%) "
                    f"y_max={y.max():.1f} y_mean={y[y>0].mean():.2f}"
                    if active_cnt > 0 else
                    f"  [diag batch {_bi}] active=0/{total_cnt} ← 全零batch!"
                )

            preds, kl = model(x, fc, nZ=nZ, static=static)
            losses, occ_losses, mag_losses = [], [], []

            for occ_logit, occ_prob, loc, scale in preds:
                loss_i, occ_i, mag_i = lognormal_enn_loss(
                    y, occ_logit, loc, scale, occ_pos_weight=occ_pos_weight
                )
                losses.append(loss_i)
                occ_losses.append(occ_i)
                mag_losses.append(mag_i)

            main_loss = sum(losses) / nZ
            occ_loss  = sum(occ_losses) / nZ
            mag_loss  = sum(mag_losses) / nZ
            loss = main_loss + kl_weight * kl

            # ── 诊断：loc 和真实 log1p(y) 的差距（只在 epoch 0 前3个 batch）
            if _bi < _diag_batches:
                with torch.no_grad():
                    pos_mask = y > 0
                    if pos_mask.sum() > 0:
                        avg_loc = sum(
                            loc[pos_mask] for _, _, loc, _ in preds
                        ) / nZ
                        avg_scale = sum(
                            scale[pos_mask] for _, _, _, scale in preds
                        ) / nZ
                        log_y = torch.log1p(y[pos_mask])
                        loc_err = (avg_loc - log_y).abs().mean()
                        print(
                            f"    loc_err={loc_err:.3f} "
                            f"loc_mean={avg_loc.mean():.3f} "
                            f"log_y_mean={log_y.mean():.3f} "
                            f"scale_mean={avg_scale.mean():.3f}"
                        )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            last_occ = occ_loss.item()
            last_mag = mag_loss.item()
            last_kl  = kl.item()

        sch.step()

        # ── 诊断：梯度 + 参数值 + mag_nll 详情（前5个 epoch）
        if epoch < 5:
            occ_grad = model.encoder.occurrence_head[-1].weight.grad
            loc_grad = model.encoder.loc_head[-1].weight.grad
            enc_grad = model.encoder.input_proj.weight.grad
            loc_w    = model.encoder.loc_head[-1].weight

            # mag_nll 详情：只在有 active weeks 的 batch 里计算
            mag_nll_vals = [m.item() for m in mag_losses if m.item() > 0]

            print(
                f"  [grad] "
                f"occ={occ_grad.abs().mean():.4f} "
                f"loc={loc_grad.abs().mean():.4f if loc_grad is not None else 'None'} "
                f"enc={enc_grad.abs().mean():.4f}"
            )
            print(
                f"  [loc_head_w] "
                f"mean={loc_w.abs().mean():.5f} "
                f"max={loc_w.abs().max():.5f} "
                f"(should grow from 0.01)"
            )
            if mag_nll_vals:
                print(
                    f"  [mag_nll detail] "
                    f"nonzero_count={len(mag_nll_vals)}/{len(mag_losses)} "
                    f"mean={sum(mag_nll_vals)/len(mag_nll_vals):.4f}"
                )
            else:
                print(f"  [mag_nll detail] all zero this epoch ← 还是有问题"  )

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in va_ld:
                p50, p70 = model.predict(
                    b["x"], b["future_context"],
                    M=M_val, static=b.get("static", None)
                )
                val_loss += (
                    pinball(b["y"], p50, 0.5) + pinball(b["y"], p70, 0.7)
                ).item()

        val_loss /= max(1, len(va_ld))

        # ── epoch 级 magnitude 诊断：每个 epoch 都打印
        model.eval()
        with torch.no_grad():
            loc_errs, log_y_means, loc_means = [], [], []
            for b in va_ld:
                x_v = b["x"]
                fc_v = b["future_context"]
                y_v = b["y"]
                st_v = b.get("static", None)

                occ_lb, loc_b, scale_b, phi_v = model._encode(x_v, fc_v, st_v)
                pos = y_v > 0
                if pos.sum() > 0:
                    log_y = torch.log1p(y_v[pos])
                    loc_p = loc_b[pos]
                    loc_errs.append((loc_p - log_y).abs().mean().item())
                    log_y_means.append(log_y.mean().item())
                    loc_means.append(loc_p.mean().item())

            if loc_errs:
                print(
                    f"  [mag_diag] "
                    f"loc_err={sum(loc_errs)/len(loc_errs):.3f} "
                    f"loc_mean={sum(loc_means)/len(loc_means):.3f} "
                    f"log_y_mean={sum(log_y_means)/len(log_y_means):.3f}"
                )

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_sd  = {k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch+1:3d} | "
            f"train={total_loss/max(1,len(tr_ld)):.4f} | "
            f"val={val_loss:.4f} | "
            f"occ={last_occ:.4f} | "
            f"mag_nll={last_mag:.4f} | "
            f"kl={last_kl:.4f}"
            + (" *" if improved else "")
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)

    print(f"Best val: {best_val:.4f}")


def evaluate(model, va_ld, M=200):
    all_y, all_p50, all_p70 = [], [], []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            p50, p70 = model.predict(
                b["x"], b["future_context"], M=M, static=b.get("static", None)
            )
            all_y.append(b["y"].numpy())
            all_p50.append(p50.numpy())
            all_p70.append(p70.numpy())

    y = np.concatenate(all_y)
    p50 = np.concatenate(all_p50)
    p70 = np.concatenate(all_p70)
    yt = torch.tensor(y)

    return {
        "pinball50": pinball(yt, torch.tensor(p50), 0.5).item(),
        "pinball70": pinball(yt, torch.tensor(p70), 0.7).item(),
    }


# =====================================================
# 5. Forecast / Diagnostics
# =====================================================

def generate_forecast_df(model, va_ld, M=200):
    rows = []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            x, fc, y, oos = b["x"], b["future_context"], b["y"], b["oos"]
            static = b.get("static", None)
            p50, p70 = model.predict(x, fc, M=M, static=static)

            hist_mean = (x[:, :, 0].exp() - 1).mean(dim=1, keepdim=True).clamp(min=0)
            hm50 = hist_mean.expand_as(y)
            hm70 = hm50 * 1.25

            for i in range(y.shape[0]):
                for h in range(y.shape[1]):
                    rows.append({
                        "asin": b["asin"][i],
                        "order_week": pd.to_datetime(b["target_week"][h][i]),
                        "fcst_week_index": h + 1,
                        "fbi_demand": y[i, h].item(),
                        "scot_oos": oos[i, h].item(),
                        "oos": oos[i, h].item(),
                        "oos_status": oos[i, h].item(),
                        "p50_amxl": p50[i, h].item(),
                        "p70_amxl": p70[i, h].item(),
                        "p50_scot": hm50[i, h].item(),
                        "p70_scot": hm70[i, h].item(),
                    })
    return pd.DataFrame(rows)


def generate_diagnostic_df(model, va_ld, M=200, threshold=0.5):
    rows = []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            p50, p70 = model.predict(
                b["x"], b["future_context"], M=M, static=b.get("static", None)
            )
            for i in range(b["y"].shape[0]):
                for h in range(b["y"].shape[1]):
                    y_val   = b["y"][i, h].item()
                    p50_val = p50[i, h].item()
                    p70_val = p70[i, h].item()
                    rows.append({
                        "asin": b["asin"][i],
                        "order_week": pd.to_datetime(b["target_week"][h][i]),
                        "horizon": h + 1,
                        "y": y_val,
                        "p50": p50_val,
                        "p70": p70_val,
                        "true_active": int(y_val > 0),
                        "pred_active_p50": int(p50_val > threshold),
                        "pred_active_p70": int(p70_val > threshold),
                    })
    return pd.DataFrame(rows)


def underbias_diagnosis(diag_df, pred_col="p70", threshold=0.5):
    y    = diag_df["y"].values
    pred = diag_df[pred_col].values

    true_active = y > 0
    pred_active = pred > threshold

    tp = np.sum(true_active & pred_active)
    fp = np.sum(~true_active & pred_active)
    fn = np.sum(true_active & ~pred_active)
    tn = np.sum(~true_active & ~pred_active)

    recall    = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1        = 2 * precision * recall / max(1e-8, precision + recall)

    total_under = np.maximum(y - pred, 0).sum()

    missed_active_mask    = true_active & ~pred_active
    magnitude_under_mask  = true_active & pred_active

    missed_under    = np.maximum(y[missed_active_mask]   - pred[missed_active_mask],   0).sum()
    magnitude_under = np.maximum(y[magnitude_under_mask] - pred[magnitude_under_mask], 0).sum()

    ratio = np.array([np.nan])
    if magnitude_under_mask.sum() > 0:
        ratio = pred[magnitude_under_mask] / np.maximum(y[magnitude_under_mask], 1e-8)

    return pd.DataFrame([{
        "pred_col": pred_col,
        "threshold": threshold,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "occurrence_recall": recall,
        "occurrence_precision": precision,
        "occurrence_f1": f1,
        "total_underbias": total_under,
        "underbias_rate": total_under / max(1e-8, y.sum()),
        "missed_active_share": missed_under / max(1e-8, total_under),
        "magnitude_under_share": magnitude_under / max(1e-8, total_under),
        "avg_pred_over_true_when_active_predicted": np.nanmean(ratio),
        "median_pred_over_true_when_active_predicted": np.nanmedian(ratio),
    }])


def magnitude_gap(diag_df):
    df = diag_df[diag_df["true_active"] == 1].copy()
    if len(df) == 0:
        print("No active weeks.")
        return pd.DataFrame()

    y, p50, p70 = df["y"].values, df["p50"].values, df["p70"].values
    out = pd.DataFrame([{
        "true_active_mean": y.mean(),
        "p50_active_mean":  p50.mean(),
        "p70_active_mean":  p70.mean(),
        "p50_pct_of_true":  p50.mean() / max(y.mean(), 1e-8),
        "p70_pct_of_true":  p70.mean() / max(y.mean(), 1e-8),
        "p50_gap": y.mean() - p50.mean(),
        "p70_gap": y.mean() - p70.mean(),
    }])
    print("\n[Magnitude Gap - Active weeks only]")
    print(out.T)
    return out


def check_anchor_quality(data_dict, history=52, horizon=20):
    rows = []
    for asin, d in data_dict.items():
        features = d["features"]
        demand   = d["demand"]
        T = len(demand)
        start = T - history - horizon
        if start < 0:
            continue

        x = features[start:start + history]
        y = demand[start + history:start + history + horizon]

        loc_anchor = 0.40 * x[-1, 8] + 0.40 * x[-1, 9] + 0.20 * x[-1, 10]
        anchor_demand = max(float(np.expm1(loc_anchor)), 0.0)

        for h in range(horizon):
            y_val = float(y[h])
            if y_val > 0:
                rows.append({
                    "asin": asin, "horizon": h + 1,
                    "y": y_val, "anchor": anchor_demand,
                    "ratio": anchor_demand / max(y_val, 1e-8),
                    "under": max(y_val - anchor_demand, 0.0),
                    "over":  max(anchor_demand - y_val, 0.0),
                })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("ANCHOR QUALITY CHECK")
    print("=" * 80)
    if len(df) == 0:
        print("No active target weeks found.")
        return df

    print("Active rows:", len(df))
    print(df[["y","anchor","ratio","under","over"]].describe(
        percentiles=[0.5, 0.7, 0.8, 0.9, 0.95]
    ))
    print("\nAnchor mean ratio:", df["anchor"].mean() / max(df["y"].mean(), 1e-8))
    print("Anchor median ratio:", df["ratio"].median())
    print("Anchor underbias rate:", df["under"].sum() / max(df["y"].sum(), 1e-8))
    print("Anchor overbias rate:",  df["over"].sum()  / max(df["y"].sum(), 1e-8))
    return df


# =====================================================
# 6. Run
# =====================================================

def run_one_group(
    data_group,
    group_name,
    prior_scale=0.5,
    beta_peak=1.5,
    occ_enn_scale=0.15,
    loc_enn_scale=0.50,
    epochs=30,
    history=52,
    horizon=20,
    d_model=24,        # 32 → 24
    d_z=8,             # 16 → 8
    batch_size=128,    # 64 → 128
    M_eval=200,
    kl_weight=0.003,
    occ_pos_weight=1.5,  # 4.0 → 2.5
    patience=5,
):
    print("\n" + "#" * 70)
    print(f"Running group: {group_name}")
    print("#" * 70)

    data, context_dim, context_cols, n_static = load_real_data(data_group)
    all_demand = np.concatenate([d["demand"] for d in data.values()])

    print(f"ASINs: {len(data)}")
    print(f"Rows: {len(data_group)}")
    print(f"Zero rate: {(all_demand == 0).mean():.1%}")
    print(f"d_model={d_model} | d_z={d_z} | batch_size={batch_size}")
    print(f"occ_pos_weight={occ_pos_weight} | patience={patience} | weight_decay=1e-4")

    anchor_diag_df = check_anchor_quality(data, history=history, horizon=horizon)

    tr_ds = DemandDataset(data, history=history, horizon=horizon, mode="train", val_weeks=horizon)
    va_ds = DemandDataset(data, history=history, horizon=horizon, mode="val",   val_weeks=horizon)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(tr_ds)} | Val samples: {len(va_ds)}")

    model = TCN_ENN_LogNormal(
        input_dim=11,
        context_dim=context_dim,
        d_model=d_model,
        d_z=d_z,
        horizon=horizon,
        prior_scale=prior_scale,
        beta_peak=beta_peak,
        static_dim=n_static,
        occ_enn_scale=occ_enn_scale,
        loc_enn_scale=loc_enn_scale,
    )

    print(
        "Trainable params:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    train(
        model, tr_ld, va_ld,
        epochs=epochs,
        nZ=8,
        lr=5e-4,
        kl_weight=kl_weight,
        occ_pos_weight=occ_pos_weight,
        M_val=100,
        patience=patience,
    )

    eval_metrics = evaluate(model, va_ld, M=M_eval)
    print("\nPinball:")
    print(eval_metrics)

    forecast_df = generate_forecast_df(model, va_ld, M=M_eval)
    forecast_df["zero_group_run"] = group_name

    diag_df  = generate_diagnostic_df(model, va_ld, M=M_eval, threshold=0.5)
    diag_p50 = underbias_diagnosis(diag_df, pred_col="p50", threshold=0.5)
    diag_p70 = underbias_diagnosis(diag_df, pred_col="p70", threshold=0.5)
    mag_gap_df = magnitude_gap(diag_df)

    print("\nUnderbias Diagnosis - P50:")
    print(diag_p50.T)
    print("\nUnderbias Diagnosis - P70:")
    print(diag_p70.T)

    quantiles = [0.5, 0.7]
    wape_df = calculate_wape_using_lp_oos2(
        forecast_df, quantiles, remove_oos_dp=True, source="lp"
    )

    cols_p50 = ["p50_amxl_penalty","p50_scot_penalty","p50_amxl_overbias",
                "p50_scot_overbias","p50_amxl_underbias","p50_scot_underbias","fbi_demand"]
    cols_p70 = ["p70_amxl_penalty","p70_scot_penalty","p70_amxl_overbias",
                "p70_scot_overbias","p70_amxl_underbias","p70_scot_underbias","fbi_demand"]

    p50_wape, p50_penalty_diff = quick_error_check(wape_df, cols_p50)
    p70_wape, p70_penalty_diff = quick_error_check(wape_df, cols_p70)

    print("\nP50 WAPE / Penalty Summary:")
    print(p50_wape)
    print("P50 penalty diff:", p50_penalty_diff)
    print("\nP70 WAPE / Penalty Summary:")
    print(p70_wape)
    print("P70 penalty diff:", p70_penalty_diff)

    summary = {
        "zero_group":               group_name,
        "n_asins":                  data_group["asin"].nunique(),
        "n_rows_raw":               len(data_group),
        "n_forecast_rows":          len(forecast_df),
        "raw_zero_rate":            (pd.to_numeric(
                                        data_group["fbi_demand"], errors="coerce"
                                     ).fillna(0) == 0).mean(),
        "forecast_true_active_ratio": (forecast_df["fbi_demand"] > 0).mean(),
        "forecast_p50_active_ratio":  (forecast_df["p50_amxl"] > 0).mean(),
        "forecast_p70_active_ratio":  (forecast_df["p70_amxl"] > 0).mean(),
        "pinball50":        eval_metrics["pinball50"],
        "pinball70":        eval_metrics["pinball70"],
        "p50_penalty_diff": p50_penalty_diff,
        "p70_penalty_diff": p70_penalty_diff,
    }

    return {
        "group":           group_name,
        "model":           model,
        "forecast_df":     forecast_df,
        "wape_df":         wape_df,
        "diag_df":         diag_df,
        "diag_p50":        diag_p50,
        "diag_p70":        diag_p70,
        "mag_gap":         mag_gap_df,
        "anchor_diag_df":  anchor_diag_df,
        "p50_wape":        p50_wape,
        "p70_wape":        p70_wape,
        "summary":         summary,
    }


def run_high_sparse_joint_regime_experiment(
    data_raw1,
    n_asins=5000,
    zero_thresholds=(0.4, 0.7),
    prior_scale=0.5,
    beta_peak=1.5,
    occ_enn_scale=0.15,
    loc_enn_scale=0.50,
    epochs=30,
    history=52,
    horizon=20,
    d_model=24,
    d_z=8,
    batch_size=128,
    M_eval=200,
    kl_weight=0.003,
    occ_pos_weight=1.5,
    patience=5,
):
    data_small = prepare_data_sample(data_raw1, n_asins=n_asins)
    data_grouped, asin_stats = add_zero_rate_group(data_small, zero_thresholds=zero_thresholds)
    data_high = data_grouped[data_grouped["zero_group"] == "high_sparse"].copy()

    print("\n" + "=" * 80)
    print("HIGH-SPARSE JOINT-REGIME LOGNORMAL EXPERIMENT v2")
    print("=" * 80)
    print(f"High sparse ASINs: {data_high['asin'].nunique()}")
    print(f"High sparse rows : {len(data_high)}")
    print("Settings:")
    print(f"  d_model={d_model} (was 32)")
    print(f"  d_z={d_z} (was 16)")
    print(f"  batch_size={batch_size} (was 64)")
    print(f"  weight_decay=1e-4 (was 1e-5)")
    print(f"  occ_pos_weight={occ_pos_weight} (was 4.0)")
    print(f"  patience={patience}")
    print(f"  oversampling=OFF")

    result = run_one_group(
        data_high,
        group_name="high_sparse_joint_regime_lognormal_v2",
        prior_scale=prior_scale,
        beta_peak=beta_peak,
        occ_enn_scale=occ_enn_scale,
        loc_enn_scale=loc_enn_scale,
        epochs=epochs,
        history=history,
        horizon=horizon,
        d_model=d_model,
        d_z=d_z,
        batch_size=batch_size,
        M_eval=M_eval,
        kl_weight=kl_weight,
        occ_pos_weight=occ_pos_weight,
        patience=patience,
    )

    summary_df = pd.DataFrame([result["summary"]])
    print("\n" + "=" * 80)
    print("HIGH-SPARSE JOINT-REGIME SUMMARY")
    print("=" * 80)
    print(summary_df)

    return result, summary_df, asin_stats


# =====================================================
# 7. Execute
# =====================================================

high_sparse_joint_result, high_sparse_joint_summary_df, asin_zero_stats = (
    run_high_sparse_joint_regime_experiment(
        data_raw1,
        n_asins=5000,
        zero_thresholds=(0.4, 0.7),
        prior_scale=0.5,
        beta_peak=1.5,
        occ_enn_scale=0.15,
        loc_enn_scale=0.50,
        epochs=30,
        history=52,
        horizon=20,
        d_model=24,
        d_z=8,
        batch_size=128,
        M_eval=200,
        kl_weight=0.003,
        occ_pos_weight=1.5,
        patience=5,
    )
)
