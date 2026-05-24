
"""
TCN + SparseAttn + ENN
NB version on HIGH-SPARSE group only

No ZINB
No start_date cut
With auxiliary pinball calibration loss
History encoder uses enhanced temporal features with promo window and quarterly seasonality
Conditional z context uses our_price, in_stock_dph, and all holiday columns found in data
No two-head explicit occ/mag
No embedding
No KL
No positive log-MSE
With tail-aware weighted NB NLL for peak-demand magnitude learning

NB-v2 changes compared with your NB-v1 version:
- add zero-rate grouping
- run main() only on high_sparse ASINs
- optional p99 extreme-ASIN filter:
  remove ASINs whose max demand exceeds high-sparse positive-demand p99
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
# 0. Sample ASINs, no date cut
# =====================================================

def prepare_data_sample(data_raw1, n_asins=5000):
    data_raw1 = data_raw1.copy()
    data_raw1["order_week"] = pd.to_datetime(data_raw1["order_week"])

    sample_asins = np.random.choice(
        data_raw1["asin"].unique(),
        size=min(n_asins, data_raw1["asin"].nunique()),
        replace=False
    )

    data_small = data_raw1[
        data_raw1["asin"].isin(sample_asins)
    ].copy()

    print("Sample ASINs:", data_small["asin"].nunique())
    print("Sample rows:", len(data_small))

    return data_small


def add_zero_rate_group_for_nb(data_raw, zero_thresholds=(0.4, 0.7)):
    """
    Only used to select high_sparse data.
    It does not change the NB model, loss, or data construction logic.
    """
    df = data_raw.copy()

    df["fbi_demand"] = pd.to_numeric(
        df["fbi_demand"],
        errors="coerce"
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
        on="asin",
        how="left"
    )

    print("\nASIN counts by zero-rate group:")
    print(
        asin_stats.groupby("zero_group")["asin"]
        .nunique()
        .reset_index(name="n_asins")
    )

    print("\nZero-rate quantiles:")
    print(asin_stats["zero_rate"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

    return df, asin_stats


# =====================================================
# 1. Data Loading
#    Encoder: enhanced temporal history features
#    Conditional z context: our_price + in_stock_dph + all holiday columns
# =====================================================

def _safe_numeric(df, col, default=0.0):
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return df




def _rolling_mean(arr, window):
    return pd.Series(arr).rolling(window, min_periods=1).mean().values


def _rolling_max(arr, window):
    return pd.Series(arr).rolling(window, min_periods=1).max().values


def _rolling_std(arr, window):
    return pd.Series(arr).rolling(window, min_periods=2).std().fillna(0).values


def _rolling_positive_mean(arr, window):
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        vals = arr[lo:i + 1]
        vals = vals[vals > 0]
        out[i] = vals.mean() if len(vals) > 0 else 0.0
    return out


def _rolling_positive_quantile(arr, window, q):
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        vals = arr[lo:i + 1]
        vals = vals[vals > 0]
        out[i] = np.quantile(vals, q) if len(vals) > 0 else 0.0
    return out


def _zero_streak(active):
    out = np.zeros(len(active), dtype=np.float32)
    cur = 0
    for i, a in enumerate(active):
        if a > 0:
            cur = 0
        else:
            cur += 1
        out[i] = cur
    return out


def _mode_or_zero(s):
    s = s.dropna()
    if len(s) == 0:
        return 0
    return s.mode().iloc[0]


def load_real_data(data_raw):
    """
    NB-v4-E feature loader: no future holiday context, no category signal, no seasonality.

    Adds:
      - category_code embedding id
      - promotion_ratio only
      - lag-safe history features for in_stock_dph only
      - forecast-origin-safe future context overwrite for DPH columns in Dataset
    """
    holiday_cols = [c for c in data_raw.columns if c.startswith("holiday_indicator_")]

    # Keep only promotion_ratio.
    # Do not use promotion_amount / promotion_pricing_amount not used in this version.
    promo_context_cols = [
        c for c in ["promotion_ratio"]
        if c in data_raw.columns
    ]

    # Keep only in_stock_dph.
    # Do not use buy_box_dph / total_dph for now to avoid extra noise/leakage risk.
    dph_cols = [
        c for c in ["in_stock_dph"]
        if c in data_raw.columns
    ]

    # Variant B/D: do NOT include future holiday_indicator_* in future context.
    # Keep historical promo_t from holidays in the history encoder.
    context_cols = ["our_price"] + dph_cols + promo_context_cols

    base_cols = [
        "asin",
        "order_week",
        "fbi_demand",
        "scot_oos",
    ]

    # context_cols intentionally excludes future holiday indicators in this variant.
    # However, holiday_cols are still needed to construct historical promo_t.
    keep_cols = [
        c for c in base_cols + context_cols + holiday_cols
        if c in data_raw.columns
    ]
    df = data_raw[keep_cols].copy()

    df = df.rename(columns={
        "asin": "ASIN",
        "order_week": "Week",
        "fbi_demand": "Demand",
        "scot_oos": "OOS",
    })

    df["Week"] = pd.to_datetime(df["Week"])
    df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce").fillna(0).clip(lower=0)
    df["OOS"] = pd.to_numeric(df["OOS"], errors="coerce").fillna(0)

    # Variant C/D: do NOT use category_code.
    # category_id is constant, so category embedding carries no group information.
    df["category_id"] = 0
    n_categories = 1

    for c in context_cols:
        df = _safe_numeric(df, c, default=0.0)

    if "our_price" in df.columns:
        df["our_price"] = np.log1p(df["our_price"].clip(lower=0))
    else:
        df["our_price"] = 0.0

    for c in dph_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0)

    if "promotion_ratio" in df.columns:
        df["promotion_ratio"] = pd.to_numeric(df["promotion_ratio"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=5.0)
    else:
        df["promotion_ratio"] = 0.0

    for c in holiday_cols:
        df[c] = df[c].clip(lower=0, upper=1)

    df = df.sort_values(["ASIN", "Week"]).reset_index(drop=True)

    # Lag-safe DPH columns for history features.
    for c in dph_cols:
        df[c + "_lag1"] = df.groupby("ASIN")[c].shift(1).fillna(0.0)

    if len(holiday_cols) > 0:
        holiday_window = np.zeros(len(df), dtype=np.float32)
        for c in holiday_cols:
            cur = df[c].values.astype(float)
            prev_window = np.roll(cur, -1)
            prev_window[-1] = 0
            holiday_window = np.maximum(holiday_window, np.maximum(cur, prev_window))
        df["promo_t"] = holiday_window
    else:
        df["promo_t"] = 0.0

    df["t"] = ((df["Week"] - df["Week"].min()).dt.days // 7).astype(int)

    dph_context_idx = [context_cols.index(c) for c in dph_cols if c in context_cols]

    data = {}

    for asin, group in df.groupby("ASIN"):
        group = group.reset_index(drop=True)

        demand = group["Demand"].values.astype(float)
        oos = group["OOS"].values.astype(float)
        weeks = group["Week"].values
        t = group["t"].values
        T = len(demand)

        category_id = 0

        v_t = np.log1p(demand)
        b_t = (demand > 0).astype(float)

        d_t = np.zeros(T)
        last = -1
        for i in range(T):
            if b_t[i] > 0:
                last = i
            d_t[i] = (i - last) / 52.0 if last >= 0 else 1.0

        price_log = group["our_price"].values.astype(float)
        promo_ratio = group["promotion_ratio"].values.astype(float)

        def lag_arr(col):
            if col in dph_cols:
                return group[col + "_lag1"].values.astype(float)
            return np.zeros(T, dtype=float)

        in_stock_lag = lag_arr("in_stock_dph")
        # buy_box_dph and total_dph are intentionally not used in this version.

        hist_nonzero_mean_52 = _rolling_positive_mean(demand, 52)
        hist_nonzero_p75_52 = _rolling_positive_quantile(demand, 52, 0.75)
        recent_peak_13 = _rolling_max(demand, 13)

        active_rate_4 = _rolling_mean(b_t, 4)
        active_rate_13 = _rolling_mean(b_t, 13)

        oos_rate_4 = _rolling_mean(oos, 4)
        oos_rate_13 = _rolling_mean(oos, 13)

        instock_mean_4 = _rolling_mean(in_stock_lag, 4)
        instock_mean_13 = _rolling_mean(in_stock_lag, 13)

        # No buy-box / total-DPH rolling features in this version.

        zero_streak = _zero_streak(b_t) / 52.0

        positive_mean_4 = _rolling_positive_mean(demand, 4)
        positive_mean_13 = _rolling_positive_mean(demand, 13)
        positive_max_13 = _rolling_max(demand, 13)
        positive_std_13 = _rolling_std(np.log1p(demand), 13)

        promo_ratio_mean_4 = _rolling_mean(promo_ratio, 4)
        promo_ratio_mean_13 = _rolling_mean(promo_ratio, 13)

        features = np.stack([
            v_t,
            b_t,
            d_t,
            group["promo_t"].values.astype(float),

            np.log1p(hist_nonzero_mean_52),
            np.log1p(hist_nonzero_p75_52),
            np.log1p(recent_peak_13),

            np.log1p(in_stock_lag),
            oos,
            active_rate_4,
            active_rate_13,
            oos_rate_4,
            oos_rate_13,
            np.log1p(instock_mean_4),
            np.log1p(instock_mean_13),
            zero_streak,
            price_log,

            np.log1p(positive_mean_4),
            np.log1p(positive_mean_13),
            np.log1p(positive_max_13),
            positive_std_13,

            promo_ratio,
            promo_ratio_mean_4,
            promo_ratio_mean_13,
        ], axis=1).astype(np.float32)

        future_context = group[context_cols].values.astype(np.float32)

        data[asin] = {
            "features": features,
            "future_context": future_context,
            "demand": demand.astype(np.float32),
            "week": weeks,
            "oos": oos.astype(np.float32),
            "category_id": category_id,
            "dph_context_idx": dph_context_idx,
            "context_cols": context_cols,
        }

    print("History encoder dim: 24")
    print(f"Conditional z context dim: {len(context_cols)}")
    print("Conditional z context columns:")
    print(context_cols)
    print("DPH context columns with forecast-origin safe overwrite:")
    print([context_cols[i] for i in dph_context_idx])
    print(f"Category embedding n_categories: {n_categories}")

    return data, len(context_cols), context_cols, n_categories


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
                target_weeks = weeks[start + history:start + history + horizon]

                self.samples.append({
                    "x": torch.tensor(
                        features[start:start + history],
                        dtype=torch.float32
                    ),
                    "future_context": torch.tensor(
                        self._make_safe_future_context(
                            future_context,
                            start,
                            history,
                            horizon,
                            d.get("dph_context_idx", [])
                        ),
                        dtype=torch.float32
                    ),
                    "y": torch.tensor(
                        demand[start + history:start + history + horizon],
                        dtype=torch.float32
                    ),
                    "asin": asin,
                    "target_week": [str(w)[:10] for w in target_weeks],
                    "oos": torch.tensor(
                        oos[start + history:start + history + horizon],
                        dtype=torch.float32
                    ),
                    "category_id": torch.tensor(
                        d.get("category_id", 0),
                        dtype=torch.long
                    ),
                })


    def _make_safe_future_context(self, future_context, start, history, horizon, dph_context_idx):
        """
        For DPH-like future context columns, overwrite horizon values with
        last historical value at forecast origin. This prevents leakage.
        """
        fc = future_context[start + history:start + history + horizon].copy()

        if len(dph_context_idx) > 0 and history > 0:
            last_hist_idx = start + history - 1
            safe_vals = future_context[last_hist_idx, dph_context_idx].copy()
            fc[:, dph_context_idx] = safe_vals[None, :]

        return fc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# =====================================================
# 3. Encoder
# =====================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            dilation=dilation
        )

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

    def forward(self, x, b_t, peak_score):
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)

        # Sparse mask: reduce attention to zero-demand weeks,
        # unless the whole sequence is zero.
        sparse_mask = (b_t == 0) & ~(b_t == 0).all(dim=1, keepdim=True)
        scores = scores.masked_fill(sparse_mask[:, None, None, :], -1e4)

        # Peak bias.
        peak_norm = peak_score / (peak_score.max(dim=1, keepdim=True)[0] + 1e-6)
        peak_bias = self.beta_peak * peak_norm
        scores = scores + peak_bias[:, None, None, :]

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return self.norm(x + out)


class TCNSparseAttnEncoder(nn.Module):
    def __init__(self, input_dim=24, d_model=32, horizon=20):
        super().__init__()

        self.horizon = horizon
        self.input_proj = nn.Linear(input_dim, d_model)

        dilations = [1, 2, 3, 4, 8, 26, 52]

        self.convs = nn.ModuleList([
            CausalConv1d(d_model, d_model, 2, d)
            for d in dilations
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in dilations
        ])

        self.sparse_attn = SparsePeakAttention(d_model, n_heads=4, beta_peak=1.0)
        self.final_norm = nn.LayerNorm(d_model)

        self.base_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, horizon)
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, horizon)
        )

    def forward(self, x):
        b_t = x[:, :, 1]

        # Encoder input uses log1p(demand), but peak attention uses sqrt(raw demand).
        peak_score = torch.sqrt(torch.expm1(x[:, :, 0]).clamp(min=0) + 1e-6)

        h = self.input_proj(x).permute(0, 2, 1)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h) + h
            h = h.permute(0, 2, 1)
            h = norm(h)
            h = F.gelu(h)
            h = h.permute(0, 2, 1)

        h = self.sparse_attn(h.permute(0, 2, 1), b_t, peak_score)
        h_t = self.final_norm(h[:, -1, :])

        mu = F.softplus(self.base_head(h_t))
        alpha = F.softplus(self.alpha_head(h_t)) + 1e-4

        return mu, alpha, h_t


# =====================================================
# 4. Conditional z generator
# =====================================================

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
        batch_size = phi.shape[0]

        ctx = future_context.reshape(batch_size, -1)
        inp = torch.cat([phi, ctx], dim=-1)

        out = self.net(inp)
        z_mean, z_logstd = out.chunk(2, dim=-1)

        z_std = F.softplus(z_logstd) + 1e-4

        return z_mean, z_std


# =====================================================
# 5. Epinet
# =====================================================

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

        sl = self.learnable(inp).view(-1, 2 * self.horizon, self.d_z)
        sl = torch.einsum("bhd,bd->bh", sl, z)

        sp = self.prior(inp).view(-1, 2 * self.horizon, self.d_z)
        sp = torch.einsum("bhd,bd->bh", sp, z) * self.prior_scale

        out = sl + sp

        return out[:, :self.horizon], out[:, self.horizon:]


# =====================================================
# 6. Full Model
# =====================================================

class TCN_ENN(nn.Module):
    def __init__(
        self,
        input_dim=24,
        context_dim=2,
        d_model=32,
        d_z=16,
        horizon=20,
        prior_scale=0.3,
        n_categories=1,
        cat_emb_dim=8
    ):
        super().__init__()

        self.d_z = d_z
        self.horizon = horizon
        self.n_categories = n_categories
        self.cat_emb_dim = cat_emb_dim

        self.category_emb = nn.Embedding(
            num_embeddings=max(1, n_categories),
            embedding_dim=cat_emb_dim
        )

        self.cat_to_phi = nn.Sequential(
            nn.Linear(cat_emb_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.cat_mu_head = nn.Sequential(
            nn.Linear(cat_emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, horizon)
        )

        self.cat_alpha_head = nn.Sequential(
            nn.Linear(cat_emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, horizon)
        )

        self.encoder = TCNSparseAttnEncoder(
            input_dim=input_dim,
            d_model=d_model,
            horizon=horizon
        )

        self.z_generator = ContextZGenerator(
            d_phi=d_model,
            context_dim=context_dim,
            d_z=d_z,
            horizon=horizon
        )

        self.epinet = Epinet(
            d_phi=d_model,
            d_z=d_z,
            horizon=horizon,
            prior_scale=prior_scale
        )

    def forward(self, x, future_context, category_id=None, nZ=8):
        mu_base, alpha_base, h_t = self.encoder(x)

        if category_id is None:
            category_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        cat_e = self.category_emb(category_id.clamp(min=0, max=self.n_categories - 1))
        cat_phi = self.cat_to_phi(cat_e)

        mu_base = mu_base + self.cat_mu_head(cat_e)
        alpha_base = alpha_base + self.cat_alpha_head(cat_e)

        # Keep the standard ENN style:
        # epinet uses a detached representation from the base encoder.
        phi = (h_t + cat_phi).detach()

        z_mean, z_std = self.z_generator(phi, future_context)

        preds = []

        for _ in range(nZ):
            eps = torch.randn_like(z_mean)
            z = z_mean + z_std * eps

            mu_e, al_e = self.epinet(phi, z)

            mu = F.softplus(mu_base + mu_e)
            alpha = F.softplus(alpha_base + al_e) + 1e-4

            preds.append((mu, alpha))

        return preds

    def predict(self, x, future_context, category_id=None, M=50):
        self.eval()

        with torch.no_grad():
            mu_base, alpha_base, h_t = self.encoder(x)

            if category_id is None:
                category_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

            cat_e = self.category_emb(category_id.clamp(min=0, max=self.n_categories - 1))
            cat_phi = self.cat_to_phi(cat_e)

            mu_base = mu_base + self.cat_mu_head(cat_e)
            alpha_base = alpha_base + self.cat_alpha_head(cat_e)

            phi = (h_t + cat_phi).detach()

            z_mean, z_std = self.z_generator(phi, future_context)

            samples = []

            for _ in range(M):
                eps = torch.randn_like(z_mean)
                z = z_mean + z_std * eps

                mu_e, al_e = self.epinet(phi, z)

                mu = F.softplus(mu_base + mu_e)
                alpha = F.softplus(alpha_base + al_e) + 1e-4

                dist = torch.distributions.NegativeBinomial(
                    total_count=(1.0 / alpha).clamp(min=1e-4),
                    probs=(mu * alpha / (1 + mu * alpha)).clamp(1e-6, 1 - 1e-6),
                )

                samples.append(dist.sample().float())

            samples = torch.stack(samples, dim=1)

            p50 = samples.quantile(0.5, dim=1)
            p70 = samples.quantile(0.7, dim=1)
            p70 = torch.maximum(p70, p50)

        return p50, p70


# =====================================================
# 7. Loss & Metrics
# =====================================================

def negbin_nll_elementwise(y, mu, alpha):
    eps = 1e-6

    r = (1.0 / alpha).clamp(min=eps)
    p = (mu * alpha / (1 + mu * alpha)).clamp(eps, 1 - eps)

    nll = -(
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1)
        + r * torch.log(1 - p)
        + y * torch.log(p)
    )

    return nll


def negbin_nll(y, mu, alpha):
    return negbin_nll_elementwise(y, mu, alpha).mean()


def tail_weighted_negbin_nll(y, mu, alpha, beta_tail=0.5):
    nll = negbin_nll_elementwise(y, mu, alpha)

    # Tail-aware weight: larger positive demand gets larger training weight.
    weight = 1.0 + beta_tail * torch.log1p(y)

    return (nll * weight).sum() / weight.sum().clamp(min=1.0)


def pinball(y, pred, q):
    d = y - pred
    return torch.mean(torch.max(q * d, (q - 1) * d))


# =====================================================
# 8. Training
# =====================================================

def train(
    model,
    tr_ld,
    va_ld,
    epochs=10,
    nZ=8,
    lr=1e-3,
    lambda_q=0.05,
    beta_tail=0.5
):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    best_sd = None

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0

        for b in tr_ld:
            x = b["x"]
            future_context = b["future_context"]
            y = b["y"]

            category_id = b.get("category_id", None)
            preds = model(x, future_context, category_id=category_id, nZ=nZ)

            nll_loss = sum(
                tail_weighted_negbin_nll(y, mu, alpha, beta_tail=beta_tail)
                for mu, alpha in preds
            ) / nZ

            mu_stack = torch.stack(
                [mu for mu, alpha in preds],
                dim=1
            )

            p50_train = mu_stack.quantile(0.5, dim=1)
            p70_train = mu_stack.quantile(0.7, dim=1)
            p70_train = torch.maximum(p70_train, p50_train)

            q_loss = (
                pinball(y, p50_train, 0.5)
                +
                pinball(y, p70_train, 0.7)
            )

            loss = nll_loss + lambda_q * q_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item()

        sch.step()

        model.eval()
        vl = 0.0

        with torch.no_grad():
            for b in va_ld:
                p50, p70 = model.predict(
                    b["x"],
                    b["future_context"],
                    category_id=b.get("category_id", None),
                    M=50
                )

                vl += (
                    pinball(b["y"], p50, 0.5)
                    +
                    pinball(b["y"], p70, 0.7)
                ).item()

        vl /= max(1, len(va_ld))

        if vl < best_val:
            best_val = vl
            best_sd = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1:3d} | "
            f"train={tr_loss / max(1, len(tr_ld)):.4f} | "
            f"val={vl:.4f} | tail_beta={beta_tail}"
        )

    if best_sd is not None:
        model.load_state_dict(best_sd)

    print(f"Best val: {best_val:.4f}")


# =====================================================
# 9. Evaluation
# =====================================================

def evaluate(model, va_ld, M=100):
    all_y = []
    all_p50 = []
    all_p70 = []

    model.eval()

    with torch.no_grad():
        for b in va_ld:
            p50, p70 = model.predict(
                b["x"],
                b["future_context"],
                category_id=b.get("category_id", None),
                M=M
            )

            all_y.append(b["y"].numpy())
            all_p50.append(p50.numpy())
            all_p70.append(p70.numpy())

    y = np.concatenate(all_y)
    p50 = np.concatenate(all_p50)
    p70 = np.concatenate(all_p70)

    yt = torch.tensor(y)

    pl50 = pinball(yt, torch.tensor(p50), 0.5).item()
    pl70 = pinball(yt, torch.tensor(p70), 0.7).item()

    return pl50, pl70


def hist_mean_baseline(va_ld):
    all_y = []
    all_hm = []

    for b in va_ld:
        y = b["y"]

        hist_mean = (
            b["x"][:, :, 0].exp() - 1
        ).mean(dim=1, keepdim=True).clamp(min=0)

        all_y.append(y)
        all_hm.append(hist_mean.expand_as(y))

    y_all = torch.cat(all_y)
    hm_all = torch.cat(all_hm)

    hm50 = pinball(y_all, hm_all, 0.5).item()
    hm70 = pinball(y_all, hm_all * 1.25, 0.7).item()

    return hm50, hm70


def generate_forecast_df(model, va_ld, M=50):
    rows = []
    model.eval()

    with torch.no_grad():
        for b in va_ld:
            x = b["x"]
            future_context = b["future_context"]
            y = b["y"]
            oos = b["oos"]

            p50, p70 = model.predict(
                x,
                future_context,
                category_id=b.get("category_id", None),
                M=M
            )

            hist_mean = (
                x[:, :, 0].exp() - 1
            ).mean(dim=1, keepdim=True).clamp(min=0)

            hm50 = hist_mean.expand_as(y)
            hm70 = hm50 * 1.25

            batch_size = y.shape[0]
            horizon = y.shape[1]

            for i in range(batch_size):
                asin_i = b["asin"][i]

                for h in range(horizon):
                    week_ih = b["target_week"][h][i]

                    rows.append({
                        "asin": asin_i,
                        "order_week": pd.to_datetime(week_ih),
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


# =====================================================
# 10. Underbias Diagnosis Utilities
# =====================================================

def generate_diagnostic_df(model, va_ld, M=100, threshold=0.5):
    rows = []
    model.eval()

    with torch.no_grad():
        for b in va_ld:
            x = b["x"]
            future_context = b["future_context"]
            y = b["y"]

            p50, p70 = model.predict(
                x,
                future_context,
                category_id=b.get("category_id", None),
                M=M
            )

            batch_size = y.shape[0]
            horizon = y.shape[1]

            for i in range(batch_size):
                asin_i = b["asin"][i]

                for h in range(horizon):
                    y_val = y[i, h].item()
                    p50_val = p50[i, h].item()
                    p70_val = p70[i, h].item()

                    rows.append({
                        "asin": asin_i,
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
    df = diag_df.copy()

    y = df["y"].values
    pred = df[pred_col].values

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

    missed_active_under = np.maximum(
        y[missed_active_mask] - pred[missed_active_mask],
        0
    ).sum()

    magnitude_under = np.maximum(
        y[magnitude_under_mask] - pred[magnitude_under_mask],
        0
    ).sum()

    missed_share = missed_active_under / max(1e-8, total_under)
    magnitude_share = magnitude_under / max(1e-8, total_under)

    avg_pred_over_true_when_active_predicted = np.nan
    median_pred_over_true_when_active_predicted = np.nan

    if np.sum(magnitude_under_mask) > 0:
        ratio = pred[magnitude_under_mask] / np.maximum(y[magnitude_under_mask], 1e-8)
        avg_pred_over_true_when_active_predicted = np.mean(ratio)
        median_pred_over_true_when_active_predicted = np.median(ratio)

    summary = {
        "pred_col": pred_col,
        "threshold": threshold,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "occurrence_recall": recall,
        "occurrence_precision": precision,
        "occurrence_f1": f1,
        "total_underbias_amount": total_under,
        "total_underbias_rate_vs_total_y": total_under / max(1e-8, total_y),
        "missed_active_under_amount": missed_active_under,
        "magnitude_under_amount": magnitude_under,
        "missed_active_under_share": missed_share,
        "magnitude_under_share": magnitude_share,
        "avg_pred_over_true_when_active_predicted": avg_pred_over_true_when_active_predicted,
        "median_pred_over_true_when_active_predicted": median_pred_over_true_when_active_predicted,
    }

    return pd.DataFrame([summary])


# =====================================================
# 11. Main
# =====================================================

def main(
    data_raw1,
    prior_scale=0.3,
    epochs=10,
    history=52,
    horizon=20,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=50,
    lambda_q=0.05,
    beta_tail=0.5
):
    print("NB-v4-E | no future holiday | no category | no seasonality | safe in_stock_dph + promotion_ratio | p99 clean high-sparse")
    print("=" * 60)

    data, context_dim, context_cols, n_categories = load_real_data(data_raw1)

    all_demand = np.concatenate([
        d["demand"]
        for d in data.values()
    ])

    print(f"ASINs: {len(data)}")
    print(f"Zero rate: {(all_demand == 0).mean():.1%}")

    tr_ld = DataLoader(
        DemandDataset(
            data,
            history=history,
            horizon=horizon,
            mode="train",
            val_weeks=horizon
        ),
        batch_size=batch_size,
        shuffle=True
    )

    va_ld = DataLoader(
        DemandDataset(
            data,
            history=history,
            horizon=horizon,
            mode="val",
            val_weeks=horizon
        ),
        batch_size=batch_size,
        shuffle=False
    )

    print(f"Train samples: {len(tr_ld.dataset)} | Val samples: {len(va_ld.dataset)}")

    model = TCN_ENN(
        input_dim=24,
        context_dim=context_dim,
        d_model=d_model,
        d_z=d_z,
        horizon=horizon,
        prior_scale=prior_scale,
        n_categories=n_categories,
        cat_emb_dim=8
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable params: {n_params:,}")
    print(f"d_model: {d_model} | d_z: {d_z} | prior_scale: {prior_scale} | lambda_q: {lambda_q} | beta_tail: {beta_tail}")
    print(f"n_categories: {n_categories} | cat_emb_dim: 8")

    print("Training...")
    train(
        model,
        tr_ld,
        va_ld,
        epochs=epochs,
        nZ=8,
        lr=1e-3,
        lambda_q=lambda_q,
        beta_tail=beta_tail
    )

    pl50, pl70 = evaluate(model, va_ld, M=100)
    hm50, hm70 = hist_mean_baseline(va_ld)

    print(f"{'Method':<20} {'Pinball-50':>12} {'Pinball-70':>12}")
    print("-" * 46)
    print(f"{'HistMean':<20} {hm50:>12.4f} {hm70:>12.4f}")
    print(f"{'ENN':<20} {pl50:>12.4f} {pl70:>12.4f}")

    print(
        f"{'ENN vs HistMean':<20} "
        f"{(hm50 - pl50) / hm50 * 100:>+11.1f}% "
        f"{(hm70 - pl70) / hm70 * 100:>+11.1f}%"
    )

    forecast_df = generate_forecast_df(model, va_ld, M=M_eval)

    print("Forecast dataframe preview:")
    print(forecast_df.head())
    print(f"Forecast rows: {len(forecast_df)}")

    return model, forecast_df, tr_ld, va_ld


# =====================================================
# 12. Run NB on High-Sparse Group Only
# =====================================================


def filter_extreme_asins_by_positive_p99_for_nb(
    data_high,
    demand_col="fbi_demand",
    asin_col="asin",
    q=0.99
):
    """
    Remove entire ASINs whose maximum positive demand exceeds the high-sparse
    positive-demand q-quantile.

    This is for fair comparison with the LogNormal p99-filter version:
    - Do NOT drop individual weeks.
    - Drop the full ASIN if it ever has an extreme positive-demand spike.
    """
    df = data_high.copy()
    df[demand_col] = pd.to_numeric(
        df[demand_col],
        errors="coerce"
    ).fillna(0).clip(lower=0)

    positive_demand = df.loc[df[demand_col] > 0, demand_col]

    if len(positive_demand) == 0:
        print("\nNo positive demand found. Extreme-ASIN filter skipped.")
        return df, pd.DataFrame(), np.nan

    extreme_cap = float(positive_demand.quantile(q))

    asin_peak = (
        df.groupby(asin_col)[demand_col]
        .max()
        .reset_index(name="asin_max_demand")
    )

    abnormal_asins = asin_peak.loc[
        asin_peak["asin_max_demand"] > extreme_cap,
        asin_col
    ]

    clean_df = df[
        ~df[asin_col].isin(abnormal_asins)
    ].copy()

    print("\n" + "=" * 80)
    print(f"NB EXTREME ASIN FILTER BY POSITIVE DEMAND P{int(q * 100)}")
    print("=" * 80)
    print(f"Positive-demand p{int(q * 100)} cap:", extreme_cap)
    print("Original high sparse ASINs:", df[asin_col].nunique())
    print("Abnormal ASINs removed:", abnormal_asins.nunique())
    print("Clean high sparse ASINs:", clean_df[asin_col].nunique())
    print("Original rows:", len(df))
    print("Clean rows:", len(clean_df))
    print("Original zero rate:", (df[demand_col] == 0).mean())
    print("Clean zero rate:", (clean_df[demand_col] == 0).mean())

    removed_stats = asin_peak[
        asin_peak[asin_col].isin(abnormal_asins)
    ].sort_values("asin_max_demand", ascending=False)

    print("\nTop removed ASIN max demand:")
    print(removed_stats.head(10))

    return clean_df, removed_stats, extreme_cap

def run_nb_high_sparse_experiment(
    data_raw1,
    n_asins=5000,
    zero_thresholds=(0.4, 0.7),
    prior_scale=0.3,
    epochs=60,
    history=52,
    horizon=20,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=50,
    lambda_q=0.05,
    beta_tail=0.5,
    remove_extreme_asins=True,
    extreme_q=0.99,
):
    # Step 1: sample ASINs
    data_small = prepare_data_sample(
        data_raw1,
        n_asins=n_asins
    )

    # Step 2: assign zero-rate group
    data_grouped, asin_zero_stats = add_zero_rate_group_for_nb(
        data_small,
        zero_thresholds=zero_thresholds
    )

    # Step 3: keep only high sparse ASINs
    data_high_sparse_nb = data_grouped[
        data_grouped["zero_group"] == "high_sparse"
    ].copy()

    removed_extreme_asin_stats = pd.DataFrame()
    extreme_cap = np.nan

    if remove_extreme_asins:
        data_high_sparse_nb, removed_extreme_asin_stats, extreme_cap = filter_extreme_asins_by_positive_p99_for_nb(
            data_high_sparse_nb,
            demand_col="fbi_demand",
            asin_col="asin",
            q=extreme_q
        )

    print("\n" + "=" * 80)
    print("NB VERSION ON HIGH-SPARSE ONLY")
    print("=" * 80)
    print("High sparse ASINs:", data_high_sparse_nb["asin"].nunique())
    print("High sparse rows :", len(data_high_sparse_nb))
    print(
        "High sparse zero rate:",
        (
            pd.to_numeric(
                data_high_sparse_nb["fbi_demand"],
                errors="coerce"
            ).fillna(0) == 0
        ).mean()
    )
    print("remove_extreme_asins:", remove_extreme_asins)
    print("extreme_q:", extreme_q)
    print("extreme_cap:", extreme_cap)

    # Step 4: run your original NB main() unchanged
    model, forecast_df, tr_ld, va_ld = main(
        data_high_sparse_nb,
        prior_scale=prior_scale,
        epochs=epochs,
        history=history,
        horizon=horizon,
        d_model=d_model,
        d_z=d_z,
        batch_size=batch_size,
        M_eval=M_eval,
        lambda_q=lambda_q,
        beta_tail=beta_tail
    )

    nb_filter_info = {
        "remove_extreme_asins": remove_extreme_asins,
        "extreme_q": extreme_q,
        "extreme_cap": extreme_cap,
        "removed_extreme_asin_stats": removed_extreme_asin_stats,
        "n_removed_extreme_asins": (
            removed_extreme_asin_stats["asin"].nunique()
            if len(removed_extreme_asin_stats) > 0 and "asin" in removed_extreme_asin_stats.columns
            else 0
        ),
    }

    return model, forecast_df, tr_ld, va_ld, data_high_sparse_nb, asin_zero_stats, nb_filter_info



def nb_magnitude_gap(forecast_df):
    """
    Active-week magnitude gap diagnosis, similar to the LogNormal version.
    """
    df = forecast_df.copy()
    active = df["fbi_demand"] > 0

    if active.sum() == 0:
        return pd.DataFrame([{
            "active_rows": 0,
            "true_active_mean": np.nan,
            "p50_active_mean": np.nan,
            "p70_active_mean": np.nan,
            "p50_pct_of_true": np.nan,
            "p70_pct_of_true": np.nan,
            "p50_gap": np.nan,
            "p70_gap": np.nan,
        }])

    y = df.loc[active, "fbi_demand"].values
    p50 = df.loc[active, "p50_amxl"].values
    p70 = df.loc[active, "p70_amxl"].values

    out = {
        "active_rows": int(active.sum()),
        "true_active_mean": float(np.mean(y)),
        "p50_active_mean": float(np.mean(p50)),
        "p70_active_mean": float(np.mean(p70)),
        "p50_pct_of_true": float(np.mean(p50) / max(1e-8, np.mean(y))),
        "p70_pct_of_true": float(np.mean(p70) / max(1e-8, np.mean(y))),
        "p50_gap": float(np.mean(y) - np.mean(p50)),
        "p70_gap": float(np.mean(y) - np.mean(p70)),
        "true_active_median": float(np.median(y)),
        "p50_active_median": float(np.median(p50)),
        "p70_active_median": float(np.median(p70)),
        "p50_coverage_active": float(np.mean(y <= p50)),
        "p70_coverage_active": float(np.mean(y <= p70)),
        "p50_zero_mean_on_true_zero": float(df.loc[~active, "p50_amxl"].mean()),
        "p70_zero_mean_on_true_zero": float(df.loc[~active, "p70_amxl"].mean()),
        "true_active_ratio": float(active.mean()),
        "p50_active_ratio": float((df["p50_amxl"] > 0.5).mean()),
        "p70_active_ratio": float((df["p70_amxl"] > 0.5).mean()),
    }

    return pd.DataFrame([out])


def nb_forecast_distribution_check(forecast_df):
    df = forecast_df.copy()

    return pd.DataFrame([{
        "n_rows": len(df),
        "n_asins": df["asin"].nunique(),
        "true_mean": df["fbi_demand"].mean(),
        "p50_mean": df["p50_amxl"].mean(),
        "p70_mean": df["p70_amxl"].mean(),
        "true_zero_rate": (df["fbi_demand"] == 0).mean(),
        "p50_zero_rate": (df["p50_amxl"] <= 0.5).mean(),
        "p70_zero_rate": (df["p70_amxl"] <= 0.5).mean(),
        "true_active_ratio": (df["fbi_demand"] > 0).mean(),
        "p50_active_ratio": (df["p50_amxl"] > 0.5).mean(),
        "p70_active_ratio": (df["p70_amxl"] > 0.5).mean(),
    }])

# =====================================================
# 13. WAPE + Underbias runner
# =====================================================

def run_wape_and_underbias(model, forecast_df, va_ld, M_diag=100, threshold=0.5):
    """
    This function assumes your notebook already has:
        calculate_wape_using_lp_oos2
        quick_error_check
    If not, WAPE will be skipped.

    NB-v3 also prints true mean vs predicted mean / active magnitude gap.
    """
    outputs = {}

    print("\n" + "=" * 80)
    print("NB-v3 FORECAST DISTRIBUTION CHECK")
    print("=" * 80)
    nb_dist_check = nb_forecast_distribution_check(forecast_df)
    print(nb_dist_check.T)

    print("\n" + "=" * 80)
    print("NB-v3 MAGNITUDE GAP - ACTIVE WEEKS ONLY")
    print("=" * 80)
    nb_mag_gap = nb_magnitude_gap(forecast_df)
    print(nb_mag_gap.T)

    outputs["nb_dist_check"] = nb_dist_check
    outputs["nb_mag_gap"] = nb_mag_gap

    if (
        "calculate_wape_using_lp_oos2" in globals()
        and "quick_error_check" in globals()
    ):
        quantiles = [0.5, 0.7]

        wape_df = calculate_wape_using_lp_oos2(
            forecast_df,
            quantiles,
            remove_oos_dp=True,
            source="lp"
        )

        cols_p50 = [
            "p50_amxl_penalty",
            "p50_scot_penalty",
            "p50_amxl_overbias",
            "p50_scot_overbias",
            "p50_amxl_underbias",
            "p50_scot_underbias",
            "fbi_demand"
        ]

        cols_p70 = [
            "p70_amxl_penalty",
            "p70_scot_penalty",
            "p70_amxl_overbias",
            "p70_scot_overbias",
            "p70_amxl_underbias",
            "p70_scot_underbias",
            "fbi_demand"
        ]

        p50_wape, p50_penalty_diff = quick_error_check(wape_df, cols_p50)
        p70_wape, p70_penalty_diff = quick_error_check(wape_df, cols_p70)

        print("P50 WAPE / Penalty Summary:")
        print(p50_wape)
        print("P50 penalty diff:", p50_penalty_diff)

        print("P70 WAPE / Penalty Summary:")
        print(p70_wape)
        print("P70 penalty diff:", p70_penalty_diff)

        outputs["wape_df"] = wape_df
        outputs["p50_wape"] = p50_wape
        outputs["p70_wape"] = p70_wape
        outputs["p50_penalty_diff"] = p50_penalty_diff
        outputs["p70_penalty_diff"] = p70_penalty_diff
    else:
        print("WAPE skipped: calculate_wape_using_lp_oos2 or quick_error_check not found.")

    diag_df = generate_diagnostic_df(
        model,
        va_ld,
        M=M_diag,
        threshold=threshold
    )

    p50_diag = underbias_diagnosis(
        diag_df,
        pred_col="p50",
        threshold=threshold
    )

    p70_diag = underbias_diagnosis(
        diag_df,
        pred_col="p70",
        threshold=threshold
    )

    print("Underbias Diagnosis - P50:")
    print(p50_diag.T)

    print("Underbias Diagnosis - P70:")
    print(p70_diag.T)

    outputs["diag_df"] = diag_df
    outputs["p50_diag"] = p50_diag
    outputs["p70_diag"] = p70_diag

    return outputs


# =====================================================
# 14. Execute
# =====================================================
# Run this after data_raw1 exists in your notebook.
#
# model, forecast_df, tr_ld, va_ld, data_high_sparse_nb, asin_zero_stats, nb_filter_info = (
#     run_nb_high_sparse_experiment(
#         data_raw1,
#         n_asins=5000,
#         zero_thresholds=(0.4, 0.7),
#         prior_scale=0.3,
#         epochs=60,
#         history=52,
#         horizon=20,
#         d_model=32,
#         d_z=16,
#         batch_size=64,
#         M_eval=50,
#         lambda_q=0.05,
#         beta_tail=0.5,
#         remove_extreme_asins=True,
#         extreme_q=0.99,
#     )
# )
#
# nb_outputs = run_wape_and_underbias(
#     model,
#     forecast_df,
#     va_ld,
#     M_diag=100,
#     threshold=0.5
# )
