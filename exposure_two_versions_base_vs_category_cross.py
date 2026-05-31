# ============================================================
# BEST CLEAN EXPOSURE PIPELINE
# Recommended current version:
#   Direct TCN exposure model + learnable anchor/source attention
#
# Main focus:
#   Improve long-horizon active AUC and reduce low/zero exposure false positives.
#
# Main entry point:
#   run_best_exposure_anchor_attention(...)
# ============================================================


# ============================================================
# Clean Exposure-Only Forecasting Script
# Target:
#   Predict future total_dph / buy_box_dph / in_stock_dph
#
# No demand model.
# No NB head.
# No ENN / z.
# No demand P50/P70.
#
# Main goal:
#   Check whether future in_stock_dph can be predicted accurately.
# ============================================================

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score


# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 1. Data sampling / filtering
# ============================================================

def prepare_data_from_sample_scot_intersection(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
):
    """
    Sample ASINs from data_raw1, then keep only ASINs also present in scot_df.
    """
    df = data_raw1.copy()
    scot = scot_df.copy()

    df["asin"] = df["asin"].astype(str)
    scot["asin"] = scot["asin"].astype(str)

    df["order_week"] = pd.to_datetime(df["order_week"])

    rng = np.random.default_rng(seed)
    unique_asins = df["asin"].dropna().unique()

    sample_asins = rng.choice(
        unique_asins,
        size=min(n_asins, len(unique_asins)),
        replace=False,
    )

    sample_asin_set = set(sample_asins)
    scot_asin_set = set(scot["asin"].dropna().unique())
    intersect_asins = sorted(sample_asin_set & scot_asin_set)

    print("\n" + "=" * 80)
    print("SAMPLE-SCOT ASIN INTERSECTION")
    print("=" * 80)
    print("Sample ASINs:", len(sample_asin_set))
    print("SCOT ASINs:", len(scot_asin_set))
    print("Intersection ASINs:", len(intersect_asins))

    data_small = df[df["asin"].isin(intersect_asins)].copy()

    return data_small


def filter_extreme_asins(data_raw, q=0.99):
    """
    Optional: remove ASINs with extremely large demand / DPH scale.
    """
    df = data_raw.copy()

    stats = (
        df.groupby("asin")
        .agg(
            max_demand=("fbi_demand", "max"),
            max_total_dph=("total_dph", "max"),
            max_buy_box_dph=("buy_box_dph", "max"),
            max_instock_dph=("in_stock_dph", "max"),
        )
        .reset_index()
    )

    thresholds = {
        "max_demand": stats["max_demand"].quantile(q),
        "max_total_dph": stats["max_total_dph"].quantile(q),
        "max_buy_box_dph": stats["max_buy_box_dph"].quantile(q),
        "max_instock_dph": stats["max_instock_dph"].quantile(q),
    }

    keep = stats[
        (stats["max_demand"] <= thresholds["max_demand"])
        & (stats["max_total_dph"] <= thresholds["max_total_dph"])
        & (stats["max_buy_box_dph"] <= thresholds["max_buy_box_dph"])
        & (stats["max_instock_dph"] <= thresholds["max_instock_dph"])
    ]["asin"]

    out = df[df["asin"].isin(set(keep))].copy()

    print("\n" + "=" * 80)
    print("EXTREME ASIN FILTER")
    print("=" * 80)
    print("Original ASINs:", df["asin"].nunique())
    print("Kept ASINs:", out["asin"].nunique())
    print("Removed ASINs:", df["asin"].nunique() - out["asin"].nunique())
    print("Thresholds:", thresholds)

    return out


# ============================================================
# 2. Feature preparation
# ============================================================

def _safe_numeric(s, fill=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def _encode_static_features(df):
    """
    Minimal static features:
      - gl_product_group
      - ind_top10_brand

    Encoded as code + frequency.
    """
    df = df.copy()
    out_cols = []

    for c in ["gl_product_group", "ind_top10_brand"]:
        if c not in df.columns:
            continue

        raw = df[c].astype(str).fillna("MISSING")

        codes, uniques = pd.factorize(raw)
        denom = max(len(uniques) - 1, 1)

        code_col = f"stock_static__{c}__code"
        freq_col = f"stock_static__{c}__freq"

        df[code_col] = codes.astype(float) / denom

        freq = raw.value_counts(normalize=True)
        df[freq_col] = raw.map(freq).fillna(0.0).astype(float)

        out_cols.extend([code_col, freq_col])

    return df, out_cols


def load_exposure_data(data_raw, dph_cap_q=0.995):
    """
    Build per-ASIN dictionary.

    History features are used by encoder.
    Future context is used by decoder.
    """
    df = data_raw.copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])
    df = df.sort_values(["asin", "order_week"]).reset_index(drop=True)

    required = [
        "asin",
        "order_week",
        "fbi_demand",
        "total_dph",
        "buy_box_dph",
        "in_stock_dph",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in ["fbi_demand", "total_dph", "buy_box_dph", "in_stock_dph"]:
        df[c] = _safe_numeric(df[c]).clip(lower=0.0)

    # Cap DPH to avoid huge outliers dominating exposure loss.
    for c in ["total_dph", "buy_box_dph", "in_stock_dph"]:
        cap = df[c].quantile(dph_cap_q)
        df[c] = df[c].clip(upper=cap)

    if "our_price" not in df.columns:
        df["our_price"] = 0.0
    df["our_price"] = _safe_numeric(df["our_price"]).clip(lower=0.0)

    if "scot_oos" not in df.columns:
        df["scot_oos"] = 0.0
    df["scot_oos"] = _safe_numeric(df["scot_oos"]).clip(0, 1)

    # Calendar / season.
    df["order_month"] = df["order_week"].dt.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * df["order_month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["order_month"] / 12.0)

    df["season_winter"] = df["order_month"].isin([12, 1, 2]).astype(float)
    df["season_spring"] = df["order_month"].isin([3, 4, 5]).astype(float)
    df["season_summer"] = df["order_month"].isin([6, 7, 8]).astype(float)
    df["season_fall"] = df["order_month"].isin([9, 10, 11]).astype(float)

    # Static features.
    df, static_cols = _encode_static_features(df)

    holiday_cols = [c for c in df.columns if c.startswith("holiday_indicator_")]
    distance_cols = [c for c in df.columns if c.startswith("distance_")]

    for c in holiday_cols + distance_cols:
        df[c] = _safe_numeric(df[c])

    context_cols = (
        ["our_price"]
        + holiday_cols
        + distance_cols
        + [
            "order_month",
            "month_sin",
            "month_cos",
            "season_winter",
            "season_spring",
            "season_summer",
            "season_fall",
        ]
        + static_cols
        + [
            # Placeholders filled at each forecast origin.
            "hist_total_dph_last_log",
            "hist_total_dph_mean4_log",
            "hist_total_dph_mean13_log",
            "hist_buy_box_dph_last_log",
            "hist_buy_box_dph_mean4_log",
            "hist_buy_box_dph_mean13_log",
            "hist_instock_dph_last_log",
            "hist_instock_dph_mean4_log",
            "hist_instock_dph_mean13_log",
        ]
    )

    context_cols = list(dict.fromkeys(context_cols))

    # Add placeholder columns.
    for c in context_cols:
        if c not in df.columns:
            df[c] = 0.0

    data = {}

    for asin, g in df.groupby("asin"):
        g = g.sort_values("order_week").reset_index(drop=True)

        demand = g["fbi_demand"].values.astype(np.float32)
        total = g["total_dph"].values.astype(np.float32)
        buy = g["buy_box_dph"].values.astype(np.float32)
        instock = g["in_stock_dph"].values.astype(np.float32)
        price = g["our_price"].values.astype(np.float32)
        oos = g["scot_oos"].values.astype(np.float32)

        # History encoder features.
        # Keep it compact and exposure-focused.
        week_idx = np.arange(len(g))
        week_sin = np.sin(2 * np.pi * week_idx / 52.0)
        week_cos = np.cos(2 * np.pi * week_idx / 52.0)

        features = np.stack(
            [
                np.log1p(demand),
                (demand > 0).astype(float),
                np.log1p(total),
                np.log1p(buy),
                np.log1p(instock),
                price,
                oos,
                week_sin,
                week_cos,
            ],
            axis=1,
        ).astype(np.float32)

        # Normalize price per ASIN roughly.
        if np.std(features[:, 5]) > 1e-8:
            features[:, 5] = (features[:, 5] - features[:, 5].mean()) / (features[:, 5].std() + 1e-8)

        data[asin] = {
            "week": g["order_week"].values,
            "features": features,
            "demand": demand,
            "total_dph": total,
            "buy_box_dph": buy,
            "in_stock_dph": instock,
            "future_context": g[context_cols].values.astype(np.float32),
            "context_cols": context_cols,
        }

    print("\n" + "=" * 80)
    print("EXPOSURE DATA LOADED")
    print("=" * 80)
    print("ASINs:", len(data))
    print("Context dim:", len(context_cols))
    print("Static cols:", [c for c in context_cols if c.startswith("stock_static__")])
    print("Holiday cols:", len(holiday_cols))
    print("Distance cols:", len(distance_cols))

    return data, len(context_cols), context_cols


# ============================================================
# 3. Dataset
# ============================================================

class ExposureDataset(Dataset):
    def __init__(self, data, history=52, horizon=20, mode="train", val_weeks=20):
        self.samples = []
        self.data = data
        self.history = history
        self.horizon = horizon

        for asin, d in data.items():
            T = len(d["features"])

            if mode == "train":
                starts = range(max(0, T - val_weeks - horizon - history + 1))
            else:
                s = T - history - horizon
                starts = [s] if s >= 0 else []

            for start in starts:
                self.samples.append((asin, start))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _hist_mean(arr, end, window):
        x = arr[max(0, end - window):end]
        if len(x) == 0:
            return 0.0
        return float(np.mean(x))

    def _make_future_context(self, d, start):
        h = self.history
        H = self.horizon
        fc = d["future_context"][start+h:start+h+H].copy()
        cols = d["context_cols"]
        idx = {c: i for i, c in enumerate(cols)}

        end = start + h

        total = d["total_dph"]
        buy = d["buy_box_dph"]
        instock = d["in_stock_dph"]

        vals = {
            "hist_total_dph_last_log": np.log1p(total[end - 1]) if end > 0 else 0.0,
            "hist_total_dph_mean4_log": np.log1p(self._hist_mean(total, end, 4)),
            "hist_total_dph_mean13_log": np.log1p(self._hist_mean(total, end, 13)),

            "hist_buy_box_dph_last_log": np.log1p(buy[end - 1]) if end > 0 else 0.0,
            "hist_buy_box_dph_mean4_log": np.log1p(self._hist_mean(buy, end, 4)),
            "hist_buy_box_dph_mean13_log": np.log1p(self._hist_mean(buy, end, 13)),

            "hist_instock_dph_last_log": np.log1p(instock[end - 1]) if end > 0 else 0.0,
            "hist_instock_dph_mean4_log": np.log1p(self._hist_mean(instock, end, 4)),
            "hist_instock_dph_mean13_log": np.log1p(self._hist_mean(instock, end, 13)),
        }

        for c, v in vals.items():
            if c in idx:
                fc[:, idx[c]] = v

        return fc

    def __getitem__(self, i):
        asin, start = self.samples[i]
        d = self.data[asin]
        h = self.history
        H = self.horizon

        return {
            "asin": asin,
            "target_week": [str(w)[:10] for w in d["week"][start+h:start+h+H]],

            "x": torch.tensor(d["features"][start:start+h], dtype=torch.float32),
            "future_context": torch.tensor(self._make_future_context(d, start), dtype=torch.float32),

            "future_total_dph": torch.tensor(d["total_dph"][start+h:start+h+H], dtype=torch.float32),
            "future_buy_box_dph": torch.tensor(d["buy_box_dph"][start+h:start+h+H], dtype=torch.float32),
            "future_instock_dph": torch.tensor(d["in_stock_dph"][start+h:start+h+H], dtype=torch.float32),

            # Only for diagnostics.
            "future_demand": torch.tensor(d["demand"][start+h:start+h+H], dtype=torch.float32),
        }


# ============================================================
# 4. Model
# ============================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class HistoryEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        dilations = [1, 2, 4, 8, 13, 26]
        self.convs = nn.ModuleList([
            CausalConv1d(d_model, d_model, kernel_size=2, dilation=d)
            for d in dilations
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in dilations])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B,T,F]
        h = self.input_proj(x).transpose(1, 2)  # [B,D,T]

        for conv, norm in zip(self.convs, self.norms):
            z = conv(h)
            h = h + z
            h = h.transpose(1, 2)
            h = norm(h)
            h = F.gelu(h)
            h = h.transpose(1, 2)

        h = h.transpose(1, 2)  # [B,T,D]
        h_t = self.final_norm(h[:, -1, :])
        return h_t


class HorizonTCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.10):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B,H,D]
        res = x
        z = x.transpose(1, 2)
        z = F.relu(self.conv1(z))
        z = self.dropout(z)
        z = F.relu(self.conv2(z))
        z = self.dropout(z)
        z = z.transpose(1, 2)

        if z.shape[1] != res.shape[1]:
            m = min(z.shape[1], res.shape[1])
            z = z[:, :m, :]
            res = res[:, :m, :]

        return self.norm(res + z)


class ExposureTCNDecoder(nn.Module):
    def __init__(self, d_model, context_dim, horizon=20, hidden=96, dropout=0.10):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Sequential(
            nn.Linear(d_model + context_dim + 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.tcn = nn.ModuleList([
            HorizonTCNBlock(hidden, dilation=1, dropout=dropout),
            HorizonTCNBlock(hidden, dilation=2, dropout=dropout),
            HorizonTCNBlock(hidden, dilation=4, dropout=dropout),
        ])

        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden, 3),
        )

    def forward(self, h_t, future_context):
        B, H, C = future_context.shape

        h_rep = h_t.unsqueeze(1).expand(B, H, h_t.shape[-1])

        horizon_idx = torch.arange(H, device=future_context.device).float()
        horizon_idx = horizon_idx.view(1, H, 1).expand(B, H, 1) / max(H, 1)

        hsin = torch.sin(2 * np.pi * horizon_idx)
        hcos = torch.cos(2 * np.pi * horizon_idx)

        x = torch.cat([h_rep, future_context, hsin, hcos], dim=-1)

        z = self.input_proj(x)

        for block in self.tcn:
            z = block(z)

        # Predict log1p levels.
        out = F.softplus(self.out(z))
        return out


class ExposureForecastModel(nn.Module):
    def __init__(self, input_dim, context_dim, d_model=64, horizon=20):
        super().__init__()
        self.encoder = HistoryEncoder(input_dim=input_dim, d_model=d_model)
        self.decoder = ExposureTCNDecoder(
            d_model=d_model,
            context_dim=context_dim,
            horizon=horizon,
            hidden=max(96, d_model * 2),
        )

    def forward(self, x, future_context):
        h_t = self.encoder(x)
        log_hat = self.decoder(h_t, future_context)
        return log_hat


# ============================================================
# 5. Loss / training
# ============================================================

def exposure_loss(
    log_hat,
    true_total,
    true_buy,
    true_instock,
    w_total=0.30,
    w_buy=0.60,
    w_instock=1.00,
    mean_weight=0.20,
):
    true = torch.stack(
        [
            true_total.clamp(min=0),
            true_buy.clamp(min=0),
            true_instock.clamp(min=0),
        ],
        dim=-1,
    )
    target_log = torch.log1p(true)

    weights = torch.tensor(
        [w_total, w_buy, w_instock],
        dtype=log_hat.dtype,
        device=log_hat.device,
    ).view(1, 1, 3)

    point = F.huber_loss(log_hat, target_log, delta=1.0, reduction="none")
    point = (point * weights).mean()

    pred_level = torch.expm1(log_hat).clamp(min=0)

    mean_pred = torch.log1p(pred_level.mean(dim=(0, 1)))
    mean_true = torch.log1p(true.mean(dim=(0, 1)))
    mean_loss = torch.mean(torch.abs(mean_pred - mean_true) * weights.view(3))

    return point + mean_weight * mean_loss


def train_exposure_model(
    model,
    tr_ld,
    va_ld,
    epochs=60,
    lr=1e-3,
    patience=8,
    w_total=0.30,
    w_buy=0.60,
    w_instock=1.00,
    mean_weight=0.20,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best_val = float("inf")
    best_sd = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0

        for b in tr_ld:
            log_hat = model(b["x"], b["future_context"])

            loss = exposure_loss(
                log_hat,
                b["future_total_dph"],
                b["future_buy_box_dph"],
                b["future_instock_dph"],
                w_total=w_total,
                w_buy=w_buy,
                w_instock=w_instock,
                mean_weight=mean_weight,
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_sum += loss.item() * b["x"].shape[0]
            tr_n += b["x"].shape[0]

        sch.step()

        model.eval()
        va_sum, va_n = 0.0, 0

        with torch.no_grad():
            for b in va_ld:
                log_hat = model(b["x"], b["future_context"])
                loss = exposure_loss(
                    log_hat,
                    b["future_total_dph"],
                    b["future_buy_box_dph"],
                    b["future_instock_dph"],
                    w_total=w_total,
                    w_buy=w_buy,
                    w_instock=w_instock,
                    mean_weight=mean_weight,
                )
                va_sum += loss.item() * b["x"].shape[0]
                va_n += b["x"].shape[0]

        tr_loss = tr_sum / max(tr_n, 1)
        va_loss = va_sum / max(va_n, 1)

        print(f"Epoch {epoch+1:03d} | train={tr_loss:.5f} | val={va_loss:.5f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stop at epoch {epoch+1}. Best val={best_val:.5f}")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)

    return model


# ============================================================
# 6. Evaluation
# ============================================================

def predict_exposure(model, va_ld):
    rows = []
    model.eval()

    with torch.no_grad():
        for b in va_ld:
            log_hat = model(b["x"], b["future_context"])
            pred = torch.expm1(log_hat).clamp(min=0).cpu().numpy()

            B, H = b["future_instock_dph"].shape

            for i in range(B):
                for h in range(H):
                    rows.append(
                        {
                            "asin": b["asin"][i],
                            "order_week": pd.to_datetime(b["target_week"][h][i]),
                            "horizon": h + 1,

                            "true_total_dph": b["future_total_dph"][i, h].item(),
                            "pred_total_dph": pred[i, h, 0],

                            "true_buy_box_dph": b["future_buy_box_dph"][i, h].item(),
                            "pred_buy_box_dph": pred[i, h, 1],

                            "true_instock_dph": b["future_instock_dph"][i, h].item(),
                            "pred_instock_dph": pred[i, h, 2],

                            "true_demand": b["future_demand"][i, h].item(),
                        }
                    )

    return pd.DataFrame(rows)


def _wape(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return np.sum(np.abs(y - p)) / (np.sum(np.abs(y)) + 1e-8)


def _corr(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if np.std(y) < 1e-8 or np.std(p) < 1e-8:
        return np.nan
    return np.corrcoef(y, p)[0, 1]


def _auc(y_binary, score):
    try:
        if len(np.unique(y_binary)) < 2:
            return np.nan
        return roc_auc_score(y_binary, score)
    except Exception:
        return np.nan


def exposure_metrics(pred_df, prefix="pred"):
    rows = []

    specs = [
        ("total_dph", "true_total_dph", f"{prefix}_total_dph"),
        ("buy_box_dph", "true_buy_box_dph", f"{prefix}_buy_box_dph"),
        ("in_stock_dph", "true_instock_dph", f"{prefix}_instock_dph"),
    ]

    for name, true_col, pred_col in specs:
        y = pred_df[true_col].values
        p = pred_df[pred_col].values

        rows.append(
            {
                "target": name,
                "true_mean": np.mean(y),
                "pred_mean": np.mean(p),
                "pred_true_ratio": np.mean(p) / (np.mean(y) + 1e-8),
                "WAPE": _wape(y, p),
                "corr": _corr(y, p),
                "active_AUC": _auc((y > 0).astype(int), p),
                "zero_rate_true": np.mean(y <= 0),
            }
        )

    return pd.DataFrame(rows)


def add_naive_baselines_from_loader(pred_df, va_ld, context_cols):
    """
    Add naive_last / naive_mean4 / naive_mean13 predictions from historical anchor columns.
    """
    idx = {c: i for i, c in enumerate(context_cols)}

    modes = {
        "last": {
            "total": "hist_total_dph_last_log",
            "buy": "hist_buy_box_dph_last_log",
            "instock": "hist_instock_dph_last_log",
        },
        "mean4": {
            "total": "hist_total_dph_mean4_log",
            "buy": "hist_buy_box_dph_mean4_log",
            "instock": "hist_instock_dph_mean4_log",
        },
        "mean13": {
            "total": "hist_total_dph_mean13_log",
            "buy": "hist_buy_box_dph_mean13_log",
            "instock": "hist_instock_dph_mean13_log",
        },
    }

    rows = []

    for b in va_ld:
        fc = b["future_context"].numpy()
        B, H, _ = fc.shape

        for i in range(B):
            for h in range(H):
                row = {
                    "asin": b["asin"][i],
                    "order_week": pd.to_datetime(b["target_week"][h][i]),
                    "horizon": h + 1,
                }

                for mode, cols in modes.items():
                    row[f"pred_total_dph_{mode}"] = np.expm1(fc[i, h, idx[cols["total"]]])
                    row[f"pred_buy_box_dph_{mode}"] = np.expm1(fc[i, h, idx[cols["buy"]]])
                    row[f"pred_instock_dph_{mode}"] = np.expm1(fc[i, h, idx[cols["instock"]]])

                rows.append(row)

    base = pd.DataFrame(rows)
    out = pred_df.merge(base, on=["asin", "order_week", "horizon"], how="left")
    return out


def print_exposure_diagnostics(pred_df):
    print("\n" + "=" * 100)
    print("MODEL EXPOSURE METRICS")
    print("=" * 100)
    model_tbl = exposure_metrics(pred_df, prefix="pred")
    print(model_tbl.round(5).to_string(index=False))

    print("\n" + "=" * 100)
    print("MODEL VS NAIVE BASELINES")
    print("=" * 100)

    tables = []
    model_tbl2 = model_tbl.copy()
    model_tbl2.insert(0, "method", "model")
    tables.append(model_tbl2)

    # Fix:
    # Do NOT use pred_df.rename directly here because it creates duplicate
    # pred_total_dph / pred_buy_box_dph / pred_instock_dph columns.
    # Duplicate column names make pred_df[pred_col] return shape (N,2),
    # which causes broadcasting error inside WAPE.
    base_true_cols = [
        "asin",
        "order_week",
        "horizon",
        "true_total_dph",
        "true_buy_box_dph",
        "true_instock_dph",
    ]

    for mode in ["last", "mean4", "mean13"]:
        tmp = pred_df[base_true_cols].copy()

        tmp["pred_total_dph"] = pred_df[f"pred_total_dph_{mode}"].values
        tmp["pred_buy_box_dph"] = pred_df[f"pred_buy_box_dph_{mode}"].values
        tmp["pred_instock_dph"] = pred_df[f"pred_instock_dph_{mode}"].values

        tbl = exposure_metrics(tmp, prefix="pred")
        tbl.insert(0, "method", f"naive_{mode}")
        tables.append(tbl)

    comp = pd.concat(tables, ignore_index=True)
    print(comp.round(5).to_string(index=False))

    print("\nFocus on in_stock_dph:")
    print(
        comp[comp["target"] == "in_stock_dph"]
        .sort_values("WAPE")
        .round(5)
        .to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("BY HORIZON: IN_STOCK_DPH")
    print("=" * 100)

    rows = []
    for h, g in pred_df.groupby("horizon"):
        y = g["true_instock_dph"].values
        p = g["pred_instock_dph"].values
        rows.append(
            {
                "horizon": h,
                "true_mean": np.mean(y),
                "pred_mean": np.mean(p),
                "ratio": np.mean(p) / (np.mean(y) + 1e-8),
                "WAPE": _wape(y, p),
                "corr": _corr(y, p),
                "active_AUC": _auc((y > 0).astype(int), p),
            }
        )

    by_h = pd.DataFrame(rows)
    print(by_h.round(5).to_string(index=False))

    return {
        "model": model_tbl,
        "comparison": comp,
        "by_horizon_instock": by_h,
    }


# ============================================================
# 7. Main runner
# ============================================================

def run_exposure_only(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
    history=52,
    horizon=20,
    d_model=64,
    batch_size=64,
    epochs=60,
    lr=1e-3,
    patience=8,
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    w_total=0.30,
    w_buy=0.60,
    w_instock=1.00,
    mean_weight=0.20,
):
    print("=" * 100)
    print("CLEAN EXPOSURE-ONLY FORECAST")
    print("=" * 100)

    df = prepare_data_from_sample_scot_intersection(
        data_raw1=data_raw1,
        scot_df=scot_df,
        n_asins=n_asins,
        seed=seed,
    )

    if remove_extreme:
        df = filter_extreme_asins(df, q=extreme_q)

    data, context_dim, context_cols = load_exposure_data(
        df,
        dph_cap_q=dph_cap_q,
    )

    tr_ds = ExposureDataset(data, history=history, horizon=horizon, mode="train", val_weeks=horizon)
    va_ds = ExposureDataset(data, history=history, horizon=horizon, mode="val", val_weeks=horizon)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    print("\nTrain samples:", len(tr_ds))
    print("Val samples:", len(va_ds))

    input_dim = next(iter(tr_ld))["x"].shape[-1]

    model = ExposureForecastModel(
        input_dim=input_dim,
        context_dim=context_dim,
        d_model=d_model,
        horizon=horizon,
    )

    print("Input dim:", input_dim)
    print("Context dim:", context_dim)
    print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_exposure_model(
        model=model,
        tr_ld=tr_ld,
        va_ld=va_ld,
        epochs=epochs,
        lr=lr,
        patience=patience,
        w_total=w_total,
        w_buy=w_buy,
        w_instock=w_instock,
        mean_weight=mean_weight,
    )

    pred_df = predict_exposure(model, va_ld)
    pred_df = add_naive_baselines_from_loader(pred_df, va_ld, context_cols)
    diagnostics = print_exposure_diagnostics(pred_df)
    exposure_quality = diagnose_exposure_prediction_quality(pred_df)

    return {
        "model": model,
        "forecast_df": pred_df,
        "diagnostics": diagnostics,
        "exposure_quality": exposure_quality,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
        "context_cols": context_cols,
        "context_dim": context_dim,
        "data": data,
    }




# ============================================================
# 8. Strong exposure quality diagnostics
# ============================================================

def _safe_spearman(y, p):
    """
    Spearman rank correlation without requiring scipy.
    """
    y = pd.Series(np.asarray(y, dtype=float)).rank(method="average").values
    p = pd.Series(np.asarray(p, dtype=float)).rank(method="average").values

    if np.std(y) < 1e-8 or np.std(p) < 1e-8:
        return np.nan

    return float(np.corrcoef(y, p)[0, 1])


def _topk_metrics(y, p, q=0.90):
    """
    Top-k precision / recall.

    Example q=0.90:
      true top 10% exposure rows
      predicted top 10% exposure rows

    Measures whether model can identify high-exposure ASIN-week rows.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    if len(y) == 0:
        return np.nan, np.nan

    true_thr = np.quantile(y, q)
    pred_thr = np.quantile(p, q)

    true_top = y >= true_thr
    pred_top = p >= pred_thr

    precision = (true_top & pred_top).sum() / (pred_top.sum() + 1e-8)
    recall = (true_top & pred_top).sum() / (true_top.sum() + 1e-8)

    return float(precision), float(recall)


def diagnose_exposure_prediction_quality(pred_df):
    """
    Comprehensive diagnostics for exposure-only predictions.

    This is stricter than checking only pred_mean / true_mean.

    It checks:
      1. overall scale + WAPE + log_MAE + Pearson + Spearman + AUC + top-k
      2. by-horizon quality
      3. ASIN-level 20-week heterogeneity quality
      4. high/mid/low true-exposure group quality
      5. worst ASINs by in_stock_dph WAPE

    Required columns:
      asin, horizon
      true_total_dph, pred_total_dph
      true_buy_box_dph, pred_buy_box_dph
      true_instock_dph, pred_instock_dph
    """
    targets = [
        ("total_dph", "true_total_dph", "pred_total_dph"),
        ("buy_box_dph", "true_buy_box_dph", "pred_buy_box_dph"),
        ("in_stock_dph", "true_instock_dph", "pred_instock_dph"),
    ]

    print("\n" + "=" * 100)
    print("1. OVERALL EXPOSURE QUALITY")
    print("=" * 100)

    overall_rows = []

    for name, true_col, pred_col in targets:
        y = pred_df[true_col].values
        p = pred_df[pred_col].values

        top_precision, top_recall = _topk_metrics(y, p, q=0.90)

        overall_rows.append({
            "target": name,
            "true_mean": np.mean(y),
            "pred_mean": np.mean(p),
            "pred_true_ratio": np.mean(p) / (np.mean(y) + 1e-8),
            "WAPE": _wape(y, p),
            "log_MAE": np.mean(np.abs(np.log1p(y) - np.log1p(p))),
            "Pearson": _corr(y, p),
            "Spearman": _safe_spearman(y, p),
            "active_AUC": _auc((y > 0).astype(int), p),
            "top10_precision": top_precision,
            "top10_recall": top_recall,
            "zero_rate_true": np.mean(y <= 0),
            "zero_rate_pred": np.mean(p <= 1e-8),
        })

    overall_df = pd.DataFrame(overall_rows)
    print(overall_df.round(5).to_string(index=False))

    print("\n" + "=" * 100)
    print("2. BY-HORIZON QUALITY")
    print("=" * 100)

    horizon_rows = []

    for h, g in pred_df.groupby("horizon"):
        for name, true_col, pred_col in targets:
            y = g[true_col].values
            p = g[pred_col].values

            horizon_rows.append({
                "horizon": h,
                "target": name,
                "true_mean": np.mean(y),
                "pred_mean": np.mean(p),
                "ratio": np.mean(p) / (np.mean(y) + 1e-8),
                "WAPE": _wape(y, p),
                "Pearson": _corr(y, p),
                "Spearman": _safe_spearman(y, p),
                "active_AUC": _auc((y > 0).astype(int), p),
            })

    horizon_df = pd.DataFrame(horizon_rows)
    print(horizon_df.round(5).to_string(index=False))

    print("\n" + "=" * 100)
    print("3. ASIN-LEVEL HETEROGENEITY QUALITY")
    print("=" * 100)

    asin_rows = []
    asin_detail_frames = []

    for name, true_col, pred_col in targets:
        tmp_rows = []

        for asin, g in pred_df.groupby("asin"):
            y = g[true_col].values
            p = g[pred_col].values

            true_sum = y.sum()
            pred_sum = p.sum()

            tmp_rows.append({
                "asin": asin,
                "target": name,
                "asin_wape": _wape(y, p),
                "true_20wk_sum": true_sum,
                "pred_20wk_sum": pred_sum,
                "sum_error": pred_sum - true_sum,
                "sum_ratio": pred_sum / (true_sum + 1e-8),
                "active_weeks_true": int((y > 0).sum()),
                "active_weeks_pred": int((p > 1e-8).sum()),
            })

        tmp = pd.DataFrame(tmp_rows)
        asin_detail_frames.append(tmp)

        asin_rows.append({
            "target": name,
            "median_asin_wape": tmp["asin_wape"].median(),
            "p75_asin_wape": tmp["asin_wape"].quantile(0.75),
            "p90_asin_wape": tmp["asin_wape"].quantile(0.90),
            "p95_asin_wape": tmp["asin_wape"].quantile(0.95),
            "asin_sum_Pearson": _corr(tmp["true_20wk_sum"], tmp["pred_20wk_sum"]),
            "asin_sum_Spearman": _safe_spearman(tmp["true_20wk_sum"], tmp["pred_20wk_sum"]),
            "median_sum_ratio": tmp["sum_ratio"].replace([np.inf, -np.inf], np.nan).median(),
            "p10_sum_ratio": tmp["sum_ratio"].replace([np.inf, -np.inf], np.nan).quantile(0.10),
            "p90_sum_ratio": tmp["sum_ratio"].replace([np.inf, -np.inf], np.nan).quantile(0.90),
        })

    asin_df = pd.DataFrame(asin_rows)
    asin_detail_df = pd.concat(asin_detail_frames, ignore_index=True)

    print(asin_df.round(5).to_string(index=False))

    print("\n" + "=" * 100)
    print("4. HIGH / MID / LOW TRUE EXPOSURE GROUP QUALITY")
    print("=" * 100)

    group_rows = []

    for name, true_col, pred_col in targets:
        asin_sum = (
            pred_df.groupby("asin")
            .agg(
                true_sum=(true_col, "sum"),
                pred_sum=(pred_col, "sum"),
            )
            .reset_index()
        )

        # qcut can fail if many ties; rank avoids most tie problems.
        asin_sum["exposure_group"] = pd.qcut(
            asin_sum["true_sum"].rank(method="first"),
            q=3,
            labels=["low_true_exposure", "mid_true_exposure", "high_true_exposure"]
        )

        gdf = pred_df.merge(
            asin_sum[["asin", "exposure_group"]],
            on="asin",
            how="left"
        )

        for grp, g in gdf.groupby("exposure_group"):
            y = g[true_col].values
            p = g[pred_col].values

            group_rows.append({
                "target": name,
                "group": grp,
                "n_rows": len(g),
                "true_mean": np.mean(y),
                "pred_mean": np.mean(p),
                "ratio": np.mean(p) / (np.mean(y) + 1e-8),
                "WAPE": _wape(y, p),
                "Pearson": _corr(y, p),
                "Spearman": _safe_spearman(y, p),
                "active_AUC": _auc((y > 0).astype(int), p),
            })

    group_df = pd.DataFrame(group_rows)
    print(group_df.round(5).to_string(index=False))

    print("\n" + "=" * 100)
    print("5. WORST ASINS BY IN_STOCK_DPH WAPE")
    print("=" * 100)

    worst = (
        asin_detail_df[asin_detail_df["target"] == "in_stock_dph"]
        .sort_values("asin_wape", ascending=False)
        .head(30)
    )

    print(worst.round(5).to_string(index=False))

    print("\n" + "=" * 100)
    print("6. QUICK JUDGMENT")
    print("=" * 100)

    inst_overall = overall_df[overall_df["target"] == "in_stock_dph"].iloc[0]
    inst_asin = asin_df[asin_df["target"] == "in_stock_dph"].iloc[0]

    print(f"in_stock pred/true mean ratio: {inst_overall['pred_true_ratio']:.4f}")
    print(f"in_stock WAPE:                 {inst_overall['WAPE']:.4f}")
    print(f"in_stock Pearson:              {inst_overall['Pearson']:.4f}")
    print(f"in_stock Spearman:             {inst_overall['Spearman']:.4f}")
    print(f"in_stock active_AUC:           {inst_overall['active_AUC']:.4f}")
    print(f"in_stock ASIN-sum Spearman:    {inst_asin['asin_sum_Spearman']:.4f}")
    print(f"in_stock p90 ASIN WAPE:        {inst_asin['p90_asin_wape']:.4f}")

    print("\nRule of thumb:")
    print("""
Good exposure_hat should not only match mean.
Prefer:
  - pred_true_ratio close to 1
  - WAPE lower than naive_mean4 / naive_mean13
  - Pearson / Spearman clearly positive and stable by horizon
  - active_AUC high enough to separate active vs inactive exposure weeks
  - ASIN-level 20-week sum Spearman high, meaning it ranks ASIN exposure baseline correctly
  - p90/p95 ASIN WAPE not exploding
""")

    return {
        "overall": overall_df,
        "by_horizon": horizon_df,
        "asin_level": asin_df,
        "asin_detail": asin_detail_df,
        "group_level": group_df,
        "worst_asins": worst,
    }



# ============================================================
# 9. Horizon-dependent blending / shrinkage
# ============================================================

def apply_horizon_blending(
    pred_df,
    anchor="last",
    short_alpha=1.00,
    mid_alpha=0.70,
    long_alpha=0.40,
    short_end=5,
    mid_end=12,
    low_exposure_shrink=False,
    low_anchor_col="pred_instock_dph_last",
    low_hist_threshold=1e-8,
    low_shrink=0.50,
):
    """
    Blend model exposure_hat with a naive anchor using horizon-dependent trust.

    Motivation:
      Model is strong at short horizon, but long-horizon active AUC drops.
      Therefore:
        short horizon: trust model more
        long horizon: shrink toward a stable anchor

    Output columns:
      blended_total_dph
      blended_buy_box_dph
      blended_instock_dph
      alpha_h
      horizon_block
    """
    df = pred_df.copy()

    required = [
        "pred_total_dph",
        "pred_buy_box_dph",
        "pred_instock_dph",
        f"pred_total_dph_{anchor}",
        f"pred_buy_box_dph_{anchor}",
        f"pred_instock_dph_{anchor}",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for blending: {missing}. "
            "Run add_naive_baselines_from_loader first."
        )

    def alpha_func(h):
        if h <= short_end:
            return short_alpha
        elif h <= mid_end:
            return mid_alpha
        else:
            return long_alpha

    def block_func(h):
        if h <= short_end:
            return f"short_1_{short_end}"
        elif h <= mid_end:
            return f"mid_{short_end+1}_{mid_end}"
        else:
            return f"long_{mid_end+1}_20"

    df["alpha_h"] = df["horizon"].apply(alpha_func).astype(float)
    df["horizon_block"] = df["horizon"].apply(block_func)

    mapping = [
        ("total", "pred_total_dph", "blended_total_dph"),
        ("buy_box", "pred_buy_box_dph", "blended_buy_box_dph"),
        ("instock", "pred_instock_dph", "blended_instock_dph"),
    ]

    for target, pred_col, out_col in mapping:
        anchor_col = f"pred_{target}_dph_{anchor}"
        df[out_col] = (
            df["alpha_h"] * df[pred_col].astype(float)
            + (1.0 - df["alpha_h"]) * df[anchor_col].astype(float)
        )

    if low_exposure_shrink:
        if low_anchor_col not in df.columns:
            raise ValueError(f"low_anchor_col {low_anchor_col} not found.")

        low_mask = df[low_anchor_col].fillna(0.0).astype(float) <= low_hist_threshold
        df.loc[low_mask, "blended_instock_dph"] *= low_shrink

    return df


def diagnose_single_prediction_set(df, pred_prefix="pred", name="model"):
    """
    Compact metrics for one prediction set.

    pred_prefix:
      "pred"  -> pred_total_dph, pred_buy_box_dph, pred_instock_dph
      "blend" -> blend_total_dph, blend_buy_box_dph, blend_instock_dph
    """
    specs = [
        ("total_dph", "true_total_dph", f"{pred_prefix}_total_dph"),
        ("buy_box_dph", "true_buy_box_dph", f"{pred_prefix}_buy_box_dph"),
        ("in_stock_dph", "true_instock_dph", f"{pred_prefix}_instock_dph"),
    ]

    rows = []
    for target, true_col, pred_col in specs:
        y = df[true_col].values
        p = df[pred_col].values

        rows.append({
            "method": name,
            "target": target,
            "true_mean": np.mean(y),
            "pred_mean": np.mean(p),
            "ratio": np.mean(p) / (np.mean(y) + 1e-8),
            "WAPE": _wape(y, p),
            "Pearson": _corr(y, p),
            "Spearman": _safe_spearman(y, p),
            "active_AUC": _auc((y > 0).astype(int), p),
        })

    return pd.DataFrame(rows)


def evaluate_blended_exposure(
    pred_df,
    anchor="last",
    short_alpha=1.00,
    mid_alpha=0.70,
    long_alpha=0.40,
    low_exposure_shrink=False,
    low_shrink=0.50,
):
    """
    Apply blending and evaluate model vs blended predictions.
    """
    blend_df = apply_horizon_blending(
        pred_df=pred_df,
        anchor=anchor,
        short_alpha=short_alpha,
        mid_alpha=mid_alpha,
        long_alpha=long_alpha,
        low_exposure_shrink=low_exposure_shrink,
        low_shrink=low_shrink,
    )

    tmp = blend_df.copy()
    tmp["pred_total_dph"] = tmp["blended_total_dph"]
    tmp["pred_buy_box_dph"] = tmp["blended_buy_box_dph"]
    tmp["pred_instock_dph"] = tmp["blended_instock_dph"]

    print("\n" + "=" * 100)
    print("BLENDED EXPOSURE FULL DIAGNOSTIC")
    print("=" * 100)
    blended_quality = diagnose_exposure_prediction_quality(tmp)

    print("\n" + "=" * 100)
    print("MODEL VS BLENDED QUICK COMPARISON")
    print("=" * 100)

    base = diagnose_single_prediction_set(pred_df, pred_prefix="pred", name="model")

    b2 = blend_df.rename(columns={
        "blended_total_dph": "blend_total_dph",
        "blended_buy_box_dph": "blend_buy_box_dph",
        "blended_instock_dph": "blend_instock_dph",
    })
    blended = diagnose_single_prediction_set(b2, pred_prefix="blend", name="blended")

    comp = pd.concat([base, blended], ignore_index=True)
    print(comp.round(5).to_string(index=False))

    print("\nFocus on in_stock_dph:")
    print(comp[comp["target"] == "in_stock_dph"].round(5).to_string(index=False))

    return {
        "blend_df": blend_df,
        "quality": blended_quality,
        "comparison": comp,
    }


def grid_search_horizon_blending(
    pred_df,
    anchors=("last", "mean13"),
    short_alphas=(1.0,),
    mid_alphas=(0.5, 0.7, 0.9),
    long_alphas=(0.2, 0.4, 0.6, 0.8),
    low_exposure_shrink_options=(False, True),
    low_shrink_options=(0.3, 0.5, 0.7),
    sort_by="in_stock_WAPE",
):
    """
    Small grid search for horizon-dependent blending.

    Metrics returned:
      in_stock WAPE / Pearson / Spearman / AUC
      low exposure group ratio / WAPE
      p90 ASIN WAPE
      ASIN-sum Spearman
    """
    rows = []

    for anchor in anchors:
        for sa in short_alphas:
            for ma in mid_alphas:
                for la in long_alphas:
                    for use_low in low_exposure_shrink_options:
                        shrink_values = low_shrink_options if use_low else (1.0,)

                        for shrink in shrink_values:
                            try:
                                bdf = apply_horizon_blending(
                                    pred_df=pred_df,
                                    anchor=anchor,
                                    short_alpha=sa,
                                    mid_alpha=ma,
                                    long_alpha=la,
                                    low_exposure_shrink=use_low,
                                    low_shrink=shrink,
                                )
                            except Exception as e:
                                rows.append({
                                    "anchor": anchor,
                                    "short_alpha": sa,
                                    "mid_alpha": ma,
                                    "long_alpha": la,
                                    "low_exposure_shrink": use_low,
                                    "low_shrink": shrink,
                                    "error": str(e),
                                })
                                continue

                            y = bdf["true_instock_dph"].values
                            p = bdf["blended_instock_dph"].values

                            asin_sum = (
                                bdf.groupby("asin")
                                .agg(
                                    true_sum=("true_instock_dph", "sum"),
                                    pred_sum=("blended_instock_dph", "sum"),
                                )
                                .reset_index()
                            )

                            per_asin = []
                            for asin, g in bdf.groupby("asin"):
                                yy = g["true_instock_dph"].values
                                pp = g["blended_instock_dph"].values
                                per_asin.append(_wape(yy, pp))
                            per_asin = np.asarray(per_asin, dtype=float)

                            asin_group = asin_sum.copy()
                            asin_group["exposure_group"] = pd.qcut(
                                asin_group["true_sum"].rank(method="first"),
                                q=3,
                                labels=["low_true_exposure", "mid_true_exposure", "high_true_exposure"],
                            )
                            gdf = bdf.merge(
                                asin_group[["asin", "exposure_group"]],
                                on="asin",
                                how="left",
                            )
                            low = gdf[gdf["exposure_group"] == "low_true_exposure"]
                            ly = low["true_instock_dph"].values
                            lp = low["blended_instock_dph"].values

                            rows.append({
                                "anchor": anchor,
                                "short_alpha": sa,
                                "mid_alpha": ma,
                                "long_alpha": la,
                                "low_exposure_shrink": use_low,
                                "low_shrink": shrink,
                                "in_stock_ratio": np.mean(p) / (np.mean(y) + 1e-8),
                                "in_stock_WAPE": _wape(y, p),
                                "in_stock_Pearson": _corr(y, p),
                                "in_stock_Spearman": _safe_spearman(y, p),
                                "in_stock_active_AUC": _auc((y > 0).astype(int), p),
                                "asin_sum_Spearman": _safe_spearman(asin_sum["true_sum"], asin_sum["pred_sum"]),
                                "p75_asin_WAPE": np.quantile(per_asin, 0.75),
                                "p90_asin_WAPE": np.quantile(per_asin, 0.90),
                                "low_group_ratio": np.mean(lp) / (np.mean(ly) + 1e-8),
                                "low_group_WAPE": _wape(ly, lp),
                                "error": "",
                            })

    out = pd.DataFrame(rows)
    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=True)

    print("\n" + "=" * 100)
    print("HORIZON BLENDING GRID SEARCH")
    print("=" * 100)

    show_cols = [
        "anchor", "short_alpha", "mid_alpha", "long_alpha",
        "low_exposure_shrink", "low_shrink",
        "in_stock_ratio", "in_stock_WAPE", "in_stock_Pearson", "in_stock_Spearman",
        "in_stock_active_AUC", "asin_sum_Spearman",
        "p75_asin_WAPE", "p90_asin_WAPE",
        "low_group_ratio", "low_group_WAPE",
    ]
    show_cols = [c for c in show_cols if c in out.columns]

    print(out[show_cols].head(30).round(5).to_string(index=False))

    return out


def make_external_hat_df_from_blend(
    blend_df,
    use_blended=True,
    include_prob_cols=True,
):
    """
    Create exposure_hat_df to feed into demand model.

    If use_blended=True:
      pred_total_dph = blended_total_dph
      pred_buy_box_dph = blended_buy_box_dph
      pred_instock_dph = blended_instock_dph
    """
    df = blend_df.copy()

    if use_blended:
        required = ["blended_total_dph", "blended_buy_box_dph", "blended_instock_dph"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing blended columns: {missing}")

        df["pred_total_dph"] = df["blended_total_dph"]
        df["pred_buy_box_dph"] = df["blended_buy_box_dph"]
        df["pred_instock_dph"] = df["blended_instock_dph"]

    cols = [
        "asin",
        "order_week",
        "pred_total_dph",
        "pred_buy_box_dph",
        "pred_instock_dph",
    ]

    if include_prob_cols:
        for c in [
            "prob_total_active",
            "prob_buy_box_active",
            "prob_instock_active",
            "prob_demand_active",
            "alpha_h",
            "horizon_block",
        ]:
            if c in df.columns:
                cols.append(c)

    return df[cols].copy()




# ============================================================
# 10. Learnable Anchor / Source Attention Blender
# ============================================================

class AnchorAttentionDataset(Dataset):
    """
    Dataset for post-hoc learnable source attention.

    It does NOT retrain the exposure TCN.
    It learns how to combine:
        model_hat
        naive_last
        naive_mean4
        naive_mean13

    For each target:
        total_dph
        buy_box_dph
        in_stock_dph

    Final:
        final_hat = sum_k softmax(weight_k) * candidate_k

    This is a learnable version of horizon-dependent blending.
    """
    def __init__(self, pred_df, target="in_stock_dph"):
        self.df = pred_df.copy().reset_index(drop=True)
        self.target = target

        if target == "in_stock_dph":
            self.true_col = "true_instock_dph"
            self.cand_cols = [
                "pred_instock_dph",
                "pred_instock_dph_last",
                "pred_instock_dph_mean4",
                "pred_instock_dph_mean13",
            ]
            self.prob_cols = [c for c in ["prob_instock_active", "prob_demand_active"] if c in self.df.columns]
        elif target == "buy_box_dph":
            self.true_col = "true_buy_box_dph"
            self.cand_cols = [
                "pred_buy_box_dph",
                "pred_buy_box_dph_last",
                "pred_buy_box_dph_mean4",
                "pred_buy_box_dph_mean13",
            ]
            self.prob_cols = [c for c in ["prob_buy_box_active", "prob_demand_active"] if c in self.df.columns]
        elif target == "total_dph":
            self.true_col = "true_total_dph"
            self.cand_cols = [
                "pred_total_dph",
                "pred_total_dph_last",
                "pred_total_dph_mean4",
                "pred_total_dph_mean13",
            ]
            self.prob_cols = [c for c in ["prob_total_active", "prob_demand_active"] if c in self.df.columns]
        else:
            raise ValueError("target must be in_stock_dph, buy_box_dph, or total_dph")

        missing = [c for c in [self.true_col] + self.cand_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns for AnchorAttentionDataset: {missing}")

        self._build_features()

    def _build_features(self):
        df = self.df

        h = df["horizon"].astype(float).values
        h_norm = h / max(float(df["horizon"].max()), 1.0)

        # Candidate values in log space.
        cand = df[self.cand_cols].fillna(0.0).clip(lower=0.0).values.astype(np.float32)
        cand_log = np.log1p(cand)

        # Relative candidate diagnostics.
        cand_mean = cand_log.mean(axis=1, keepdims=True)
        cand_std = cand_log.std(axis=1, keepdims=True) + 1e-6
        cand_z = (cand_log - cand_mean) / cand_std

        # Horizon / block features.
        h_feat = np.stack([
            h_norm,
            np.sin(2 * np.pi * h_norm),
            np.cos(2 * np.pi * h_norm),
            (h <= 5).astype(float),
            ((h >= 6) & (h <= 12)).astype(float),
            (h >= 13).astype(float),
        ], axis=1).astype(np.float32)

        # Optional probability features.
        if len(self.prob_cols) > 0:
            prob = df[self.prob_cols].fillna(0.0).values.astype(np.float32)
        else:
            prob = np.zeros((len(df), 0), dtype=np.float32)

        # Combine features.
        X = np.concatenate([h_feat, cand_log, cand_z, prob], axis=1).astype(np.float32)

        # Standardize non-binary-ish features globally for stable training.
        self.x_mean = X.mean(axis=0, keepdims=True)
        self.x_std = X.std(axis=0, keepdims=True) + 1e-6
        Xn = (X - self.x_mean) / self.x_std

        self.X = Xn.astype(np.float32)
        self.candidates = cand.astype(np.float32)
        self.y = df[self.true_col].fillna(0.0).clip(lower=0.0).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.X[idx], dtype=torch.float32),
            "candidates": torch.tensor(self.candidates[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


class AnchorAttentionBlender(nn.Module):
    """
    Small MLP producing attention weights over candidates:
        [model_hat, naive_last, naive_mean4, naive_mean13]
    """
    def __init__(self, input_dim, hidden=64, n_candidates=4, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_candidates),
        )

    def forward(self, x, candidates, temperature=1.0):
        logits = self.net(x) / max(temperature, 1e-6)
        weights = torch.softmax(logits, dim=-1)
        pred = torch.sum(weights * candidates, dim=-1)
        return pred, weights


def _attention_blender_loss(pred, y, weights=None, candidates=None, entropy_weight=0.001):
    """
    Loss for source attention blender.

    Uses log-level Huber so both large and small ASINs matter.
    Adds tiny entropy regularization to avoid collapsed weights too early.
    """
    pred_log = torch.log1p(pred.clamp(min=0.0))
    y_log = torch.log1p(y.clamp(min=0.0))

    main = F.huber_loss(pred_log, y_log, delta=1.0, reduction="mean")

    ent = torch.tensor(0.0, device=pred.device)
    if weights is not None and entropy_weight > 0:
        ent = -torch.mean(torch.sum(weights * torch.log(weights + 1e-8), dim=-1))
        # Subtract entropy means encourage some exploration / non-collapse.
        main = main - entropy_weight * ent

    return main


def train_anchor_attention_blender_for_target(
    pred_df,
    target="in_stock_dph",
    epochs=80,
    batch_size=1024,
    lr=1e-3,
    patience=10,
    hidden=64,
    dropout=0.10,
    seed=42,
):
    """
    Train a post-hoc learnable attention blender for one target.

    Split:
      train rows: horizons 1-16 approximately and random 80%
      val rows: random 20%
    This is only for quick validation. For production, train this on training windows
    and apply to validation/test windows.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = AnchorAttentionDataset(pred_df, target=target)

    n = len(ds)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = max(int(0.2 * n), 1)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_subset = torch.utils.data.Subset(ds, train_idx.tolist())
    val_subset = torch.utils.data.Subset(ds, val_idx.tolist())

    tr_ld = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = AnchorAttentionBlender(
        input_dim=ds.X.shape[1],
        hidden=hidden,
        n_candidates=4,
        dropout=dropout,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val = float("inf")
    best_sd = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        tr_sum, tr_n = 0.0, 0

        for b in tr_ld:
            pred, w = model(b["x"], b["candidates"])
            loss = _attention_blender_loss(pred, b["y"], weights=w)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_sum += loss.item() * b["x"].shape[0]
            tr_n += b["x"].shape[0]

        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for b in va_ld:
                pred, w = model(b["x"], b["candidates"])
                loss = _attention_blender_loss(pred, b["y"], weights=w)
                va_sum += loss.item() * b["x"].shape[0]
                va_n += b["x"].shape[0]

        tr_loss = tr_sum / max(tr_n, 1)
        va_loss = va_sum / max(va_n, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{target}] Epoch {epoch+1:03d} | train={tr_loss:.5f} | val={va_loss:.5f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[{target}] Early stop at epoch {epoch+1}. Best val={best_val:.5f}")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)

    return {
        "model": model,
        "dataset": ds,
        "target": target,
        "best_val": best_val,
    }


def apply_anchor_attention_blender_for_target(pred_df, blender_result):
    """
    Apply trained anchor attention blender to one target.
    """
    target = blender_result["target"]
    model = blender_result["model"]
    ds_train = blender_result["dataset"]

    # Rebuild dataset for this pred_df.
    ds = AnchorAttentionDataset(pred_df, target=target)

    # Use train standardization to avoid leakage/drift.
    X = np.concatenate([
        np.stack([
            pred_df["horizon"].astype(float).values / max(float(pred_df["horizon"].max()), 1.0),
            np.sin(2 * np.pi * pred_df["horizon"].astype(float).values / max(float(pred_df["horizon"].max()), 1.0)),
            np.cos(2 * np.pi * pred_df["horizon"].astype(float).values / max(float(pred_df["horizon"].max()), 1.0)),
            (pred_df["horizon"].values <= 5).astype(float),
            ((pred_df["horizon"].values >= 6) & (pred_df["horizon"].values <= 12)).astype(float),
            (pred_df["horizon"].values >= 13).astype(float),
        ], axis=1).astype(np.float32),
        np.log1p(pred_df[ds.cand_cols].fillna(0.0).clip(lower=0.0).values.astype(np.float32)),
        (np.log1p(pred_df[ds.cand_cols].fillna(0.0).clip(lower=0.0).values.astype(np.float32)) -
         np.log1p(pred_df[ds.cand_cols].fillna(0.0).clip(lower=0.0).values.astype(np.float32)).mean(axis=1, keepdims=True)) /
        (np.log1p(pred_df[ds.cand_cols].fillna(0.0).clip(lower=0.0).values.astype(np.float32)).std(axis=1, keepdims=True) + 1e-6),
        pred_df[ds.prob_cols].fillna(0.0).values.astype(np.float32) if len(ds.prob_cols) > 0 else np.zeros((len(pred_df), 0), dtype=np.float32),
    ], axis=1).astype(np.float32)

    Xn = (X - ds_train.x_mean) / ds_train.x_std
    candidates = pred_df[ds.cand_cols].fillna(0.0).clip(lower=0.0).values.astype(np.float32)

    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(Xn, dtype=torch.float32)
        c_t = torch.tensor(candidates, dtype=torch.float32)
        pred, weights = model(x_t, c_t)

    out = pred_df.copy()

    if target == "in_stock_dph":
        out["attn_instock_dph"] = pred.numpy()
        prefix = "attn_instock"
    elif target == "buy_box_dph":
        out["attn_buy_box_dph"] = pred.numpy()
        prefix = "attn_buy_box"
    else:
        out["attn_total_dph"] = pred.numpy()
        prefix = "attn_total"

    w = weights.numpy()
    out[f"{prefix}_w_model"] = w[:, 0]
    out[f"{prefix}_w_last"] = w[:, 1]
    out[f"{prefix}_w_mean4"] = w[:, 2]
    out[f"{prefix}_w_mean13"] = w[:, 3]

    return out


def train_and_apply_anchor_attention_blender(
    pred_df,
    targets=("total_dph", "buy_box_dph", "in_stock_dph"),
    epochs=80,
    batch_size=1024,
    lr=1e-3,
    patience=10,
    hidden=64,
    dropout=0.10,
    seed=42,
):
    """
    Train source-attention blender for all exposure targets and apply.

    Output columns:
      attn_total_dph
      attn_buy_box_dph
      attn_instock_dph

    Weight columns:
      attn_*_w_model / last / mean4 / mean13
    """
    out_df = pred_df.copy()
    results = {}

    for target in targets:
        print("\n" + "=" * 100)
        print(f"TRAIN ANCHOR ATTENTION BLENDER: {target}")
        print("=" * 100)

        res = train_anchor_attention_blender_for_target(
            pred_df=pred_df,
            target=target,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            hidden=hidden,
            dropout=dropout,
            seed=seed,
        )
        results[target] = res

        out_df = apply_anchor_attention_blender_for_target(out_df, res)

    return {
        "attn_df": out_df,
        "blenders": results,
    }


def evaluate_anchor_attention_blender(attn_df):
    """
    Evaluate anchor attention predictions.

    It converts attention predictions into pred_* columns and uses existing diagnostics.
    """
    tmp = attn_df.copy()

    required = ["attn_total_dph", "attn_buy_box_dph", "attn_instock_dph"]
    missing = [c for c in required if c not in tmp.columns]
    if missing:
        raise ValueError(f"Missing attention prediction columns: {missing}")

    tmp["pred_total_dph"] = tmp["attn_total_dph"]
    tmp["pred_buy_box_dph"] = tmp["attn_buy_box_dph"]
    tmp["pred_instock_dph"] = tmp["attn_instock_dph"]

    print("\n" + "=" * 100)
    print("ANCHOR ATTENTION BLENDER FULL DIAGNOSTIC")
    print("=" * 100)

    quality = diagnose_exposure_prediction_quality(tmp)

    print("\n" + "=" * 100)
    print("ANCHOR ATTENTION WEIGHTS BY HORIZON BLOCK")
    print("=" * 100)

    df = attn_df.copy()
    df["horizon_block"] = pd.cut(
        df["horizon"],
        bins=[0, 5, 12, 20],
        labels=["short_1_5", "mid_6_12", "long_13_20"],
        include_lowest=True,
    )

    rows = []
    for block, g in df.groupby("horizon_block", observed=True):
        for prefix, target in [
            ("attn_total", "total_dph"),
            ("attn_buy_box", "buy_box_dph"),
            ("attn_instock", "in_stock_dph"),
        ]:
            needed = [f"{prefix}_w_model", f"{prefix}_w_last", f"{prefix}_w_mean4", f"{prefix}_w_mean13"]
            if all(c in g.columns for c in needed):
                rows.append({
                    "block": block,
                    "target": target,
                    "w_model": g[f"{prefix}_w_model"].mean(),
                    "w_last": g[f"{prefix}_w_last"].mean(),
                    "w_mean4": g[f"{prefix}_w_mean4"].mean(),
                    "w_mean13": g[f"{prefix}_w_mean13"].mean(),
                })

    weight_df = pd.DataFrame(rows)
    print(weight_df.round(4).to_string(index=False))

    print("\nFocus: long_13_20 should usually reduce w_model and increase stable anchors if attention is working.")

    return {
        "quality": quality,
        "weight_by_block": weight_df,
    }


def make_external_hat_df_from_anchor_attention(attn_df, include_prob_cols=True):
    """
    Create exposure_hat_df for demand model from anchor attention outputs.
    """
    required = ["attn_total_dph", "attn_buy_box_dph", "attn_instock_dph"]
    missing = [c for c in required if c not in attn_df.columns]
    if missing:
        raise ValueError(f"Missing attention prediction columns: {missing}")

    df = attn_df.copy()

    df["pred_total_dph"] = df["attn_total_dph"]
    df["pred_buy_box_dph"] = df["attn_buy_box_dph"]
    df["pred_instock_dph"] = df["attn_instock_dph"]

    cols = [
        "asin",
        "order_week",
        "pred_total_dph",
        "pred_buy_box_dph",
        "pred_instock_dph",
    ]

    if include_prob_cols:
        for c in [
            "prob_total_active",
            "prob_buy_box_active",
            "prob_instock_active",
            "prob_demand_active",
        ]:
            if c in df.columns:
                cols.append(c)

        # Include attention weights as confidence/explainability features if needed.
        for c in df.columns:
            if c.startswith("attn_") and "_w_" in c:
                cols.append(c)

    return df[cols].copy()


# ============================================================
# Example usage:
# ============================================================
#
# pred_df = result_exp["forecast_df"]
#
# attn_res = train_and_apply_anchor_attention_blender(
#     pred_df,
#     targets=("total_dph", "buy_box_dph", "in_stock_dph"),
#     epochs=80,
#     batch_size=1024,
#     lr=1e-3,
#     patience=10,
# )
#
# attn_df = attn_res["attn_df"]
# attn_eval = evaluate_anchor_attention_blender(attn_df)
#
# exposure_hat_for_demand = make_external_hat_df_from_anchor_attention(attn_df)
#



# ============================================================
# 11. Clean best pipeline: Direct TCN + Anchor Attention
# ============================================================

def diagnose_long_horizon_auc(attn_df, pred_prefix="attn"):
    """
    Focused diagnostic for the long-horizon AUC issue.

    pred_prefix:
      "attn" uses:
        attn_total_dph / attn_buy_box_dph / attn_instock_dph
      "pred" uses:
        pred_total_dph / pred_buy_box_dph / pred_instock_dph

    This function prints:
      1. by-horizon active AUC
      2. short/mid/long block active AUC
      3. model-vs-anchor-attention comparison if both exist
    """
    df = attn_df.copy()

    if pred_prefix == "attn":
        pred_cols = {
            "total_dph": "attn_total_dph",
            "buy_box_dph": "attn_buy_box_dph",
            "in_stock_dph": "attn_instock_dph",
        }
    else:
        pred_cols = {
            "total_dph": "pred_total_dph",
            "buy_box_dph": "pred_buy_box_dph",
            "in_stock_dph": "pred_instock_dph",
        }

    true_cols = {
        "total_dph": "true_total_dph",
        "buy_box_dph": "true_buy_box_dph",
        "in_stock_dph": "true_instock_dph",
    }

    missing = []
    for k in pred_cols:
        if pred_cols[k] not in df.columns:
            missing.append(pred_cols[k])
        if true_cols[k] not in df.columns:
            missing.append(true_cols[k])
    if missing:
        raise ValueError(f"Missing columns for long horizon AUC diagnostic: {missing}")

    print("\n" + "=" * 100)
    print("LONG-HORIZON ACTIVE AUC DIAGNOSTIC")
    print("=" * 100)

    # By horizon.
    rows = []
    for h, g in df.groupby("horizon"):
        for target in ["total_dph", "buy_box_dph", "in_stock_dph"]:
            y = (g[true_cols[target]].values > 0).astype(int)
            p = g[pred_cols[target]].values
            rows.append({
                "horizon": int(h),
                "target": target,
                "true_active_rate": y.mean(),
                "pred_mean": np.mean(p),
                "active_AUC": _auc(y, p),
                "WAPE": _wape(g[true_cols[target]].values, p),
                "Pearson": _corr(g[true_cols[target]].values, p),
                "Spearman": _safe_spearman(g[true_cols[target]].values, p),
            })

    by_h = pd.DataFrame(rows)

    print("\nBy horizon: in_stock_dph")
    print(
        by_h[by_h["target"] == "in_stock_dph"]
        .round(5)
        .to_string(index=False)
    )

    # By block.
    df["horizon_block"] = pd.cut(
        df["horizon"],
        bins=[0, 5, 12, 20],
        labels=["short_1_5", "mid_6_12", "long_13_20"],
        include_lowest=True,
    )

    block_rows = []
    for block, g in df.groupby("horizon_block", observed=True):
        for target in ["total_dph", "buy_box_dph", "in_stock_dph"]:
            y = (g[true_cols[target]].values > 0).astype(int)
            p = g[pred_cols[target]].values
            block_rows.append({
                "block": block,
                "target": target,
                "true_active_rate": y.mean(),
                "pred_mean": np.mean(p),
                "active_AUC": _auc(y, p),
                "WAPE": _wape(g[true_cols[target]].values, p),
                "Pearson": _corr(g[true_cols[target]].values, p),
                "Spearman": _safe_spearman(g[true_cols[target]].values, p),
            })

    by_block = pd.DataFrame(block_rows)

    print("\nBy horizon block")
    print(by_block.round(5).to_string(index=False))

    # Compare raw model vs attention if available.
    if all(c in df.columns for c in ["pred_instock_dph", "attn_instock_dph"]):
        comp_rows = []

        for method, col in [
            ("raw_model", "pred_instock_dph"),
            ("anchor_attention", "attn_instock_dph"),
            ("naive_last", "pred_instock_dph_last"),
            ("naive_mean13", "pred_instock_dph_mean13"),
        ]:
            if col not in df.columns:
                continue

            for block, g in df.groupby("horizon_block", observed=True):
                y_level = g["true_instock_dph"].values
                y_active = (y_level > 0).astype(int)
                p = g[col].values

                comp_rows.append({
                    "method": method,
                    "block": block,
                    "target": "in_stock_dph",
                    "active_AUC": _auc(y_active, p),
                    "WAPE": _wape(y_level, p),
                    "Pearson": _corr(y_level, p),
                    "Spearman": _safe_spearman(y_level, p),
                    "pred_true_ratio": np.mean(p) / (np.mean(y_level) + 1e-8),
                })

        comp = pd.DataFrame(comp_rows)

        print("\nRaw model vs anchor attention vs naive: in_stock_dph")
        print(comp.round(5).to_string(index=False))
    else:
        comp = None

    return {
        "by_horizon": by_h,
        "by_block": by_block,
        "comparison": comp,
    }


def run_best_exposure_anchor_attention(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
    history=52,
    horizon=20,
    d_model=64,
    batch_size=64,
    exposure_epochs=60,
    exposure_lr=1e-3,
    exposure_patience=8,
    attn_epochs=80,
    attn_batch_size=1024,
    attn_lr=1e-3,
    attn_patience=10,
    attn_hidden=64,
    attn_dropout=0.10,
    remove_extreme=True,
    extreme_q=0.99,
):
    """
    Clean best pipeline currently recommended:

      Step 1:
        Train direct TCN exposure-only model.
        This predicts:
          pred_total_dph
          pred_buy_box_dph
          pred_instock_dph

      Step 2:
        Train post-hoc anchor/source attention blender.
        It combines:
          model_hat
          naive_last
          naive_mean4
          naive_mean13

      Step 3:
        Evaluate long-horizon active AUC.
        This directly targets the long-horizon AUC problem.

      Step 4:
        Create exposure_hat_for_demand.
        This can be fed into the demand model.

    Recommended downstream:
      Use exposure_hat_for_demand as external exposure_hat input.
    """
    print("\n" + "=" * 100)
    print("BEST CLEAN PIPELINE: DIRECT TCN + ANCHOR ATTENTION")
    print("=" * 100)

    base_result = run_exposure_only(
        data_raw1=data_raw1,
        scot_df=scot_df,
        n_asins=n_asins,
        seed=seed,
        history=history,
        horizon=horizon,
        d_model=d_model,
        batch_size=batch_size,
        epochs=exposure_epochs,
        lr=exposure_lr,
        patience=exposure_patience,
        remove_extreme=remove_extreme,
        extreme_q=extreme_q,
    )

    pred_df = base_result["forecast_df"]

    print("\n" + "=" * 100)
    print("TRAIN ANCHOR/SOURCE ATTENTION BLENDER")
    print("=" * 100)

    attn_result = train_and_apply_anchor_attention_blender(
        pred_df,
        targets=("total_dph", "buy_box_dph", "in_stock_dph"),
        epochs=attn_epochs,
        batch_size=attn_batch_size,
        lr=attn_lr,
        patience=attn_patience,
        hidden=attn_hidden,
        dropout=attn_dropout,
        seed=seed,
    )

    attn_df = attn_result["attn_df"]

    attn_eval = evaluate_anchor_attention_blender(attn_df)

    long_auc_diag = diagnose_long_horizon_auc(
        attn_df,
        pred_prefix="attn",
    )

    exposure_hat_for_demand = make_external_hat_df_from_anchor_attention(
        attn_df,
        include_prob_cols=True,
    )

    print("\n" + "=" * 100)
    print("READY FOR DEMAND MODEL")
    print("=" * 100)
    print("Use this dataframe as exposure_hat_df:")
    print("    result_best['exposure_hat_for_demand']")
    print("\nColumns:")
    print(exposure_hat_for_demand.columns.tolist())

    return {
        "base_result": base_result,
        "pred_df": pred_df,
        "attn_result": attn_result,
        "attn_df": attn_df,
        "attn_eval": attn_eval,
        "long_auc_diag": long_auc_diag,
        "exposure_hat_for_demand": exposure_hat_for_demand,
    }


# ============================================================
# Clean usage:
# ============================================================
#
# result_best = run_best_exposure_anchor_attention(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     n_asins=5000,
#     seed=42,
#     history=52,
#     horizon=20,
#     d_model=64,
#     batch_size=64,
#     exposure_epochs=60,
#     exposure_lr=1e-3,
#     exposure_patience=8,
# )
#
# # Focus on long-horizon active AUC:
# long_auc = result_best["long_auc_diag"]
#
# # Final exposure hat for demand model:
# exposure_hat_for_demand = result_best["exposure_hat_for_demand"]
#



# ============================================================
# 13. FOCUSED ATTENTION REPORT / QUIET RUNNER
# ============================================================

def focused_attention_long_auc_report(attn_df):
    """
    This is the ONLY report you need for the current question:

    Question 1:
      Did anchor attention help?

    Question 2:
      Did long-horizon active_AUC improve?

    It compares:
      raw_model          = pred_instock_dph
      anchor_attention   = attn_instock_dph
      naive_last         = pred_instock_dph_last
      naive_mean13       = pred_instock_dph_mean13

    It prints:
      A. Overall in_stock comparison
      B. Long horizon h=13-20 comparison
      C. AUC by horizon
      D. Attention weights by horizon block
      E. Delta improvement raw -> attention
    """
    df = attn_df.copy()

    required = [
        "true_instock_dph",
        "pred_instock_dph",
        "attn_instock_dph",
        "pred_instock_dph_last",
        "pred_instock_dph_mean13",
        "horizon",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for focused attention report: {missing}")

    df["horizon_block"] = pd.cut(
        df["horizon"],
        bins=[0, 5, 12, 20],
        labels=["short_1_5", "mid_6_12", "long_13_20"],
        include_lowest=True,
    )

    methods = [
        ("raw_model", "pred_instock_dph"),
        ("anchor_attention", "attn_instock_dph"),
        ("naive_last", "pred_instock_dph_last"),
        ("naive_mean13", "pred_instock_dph_mean13"),
    ]

    def metrics(g, pred_col):
        y = g["true_instock_dph"].values
        p = g[pred_col].values
        return {
            "ratio": np.mean(p) / (np.mean(y) + 1e-8),
            "WAPE": _wape(y, p),
            "Pearson": _corr(y, p),
            "Spearman": _safe_spearman(y, p),
            "active_AUC": _auc((y > 0).astype(int), p),
        }

    # --------------------------------------------------------
    # A. Overall
    # --------------------------------------------------------
    overall_rows = []
    for name, col in methods:
        m = metrics(df, col)
        m["method"] = name
        overall_rows.append(m)

    overall = pd.DataFrame(overall_rows)[
        ["method", "ratio", "WAPE", "Pearson", "Spearman", "active_AUC"]
    ]

    print("\n" + "=" * 100)
    print("A. DOES ATTENTION HELP? OVERALL IN_STOCK_DPH")
    print("=" * 100)
    print(overall.round(5).to_string(index=False))

    # --------------------------------------------------------
    # B. Horizon block comparison
    # --------------------------------------------------------
    block_rows = []
    for block, g in df.groupby("horizon_block", observed=True):
        for name, col in methods:
            m = metrics(g, col)
            m["block"] = block
            m["method"] = name
            block_rows.append(m)

    block_comp = pd.DataFrame(block_rows)[
        ["block", "method", "ratio", "WAPE", "Pearson", "Spearman", "active_AUC"]
    ]

    print("\n" + "=" * 100)
    print("B. LONG-HORIZON CHECK BY BLOCK")
    print("=" * 100)
    print(block_comp.round(5).to_string(index=False))

    # --------------------------------------------------------
    # C. AUC by horizon
    # --------------------------------------------------------
    auc_rows = []
    for h, g in df.groupby("horizon"):
        row = {"horizon": int(h)}
        y = (g["true_instock_dph"].values > 0).astype(int)

        for name, col in methods:
            row[f"{name}_AUC"] = _auc(y, g[col].values)

        row["attn_minus_raw_AUC"] = row["anchor_attention_AUC"] - row["raw_model_AUC"]
        row["attn_minus_last_AUC"] = row["anchor_attention_AUC"] - row["naive_last_AUC"]
        auc_rows.append(row)

    auc_by_h = pd.DataFrame(auc_rows)

    print("\n" + "=" * 100)
    print("C. LONG-HORIZON ACTIVE_AUC BY HORIZON")
    print("=" * 100)
    print(auc_by_h.round(5).to_string(index=False))

    # --------------------------------------------------------
    # D. Attention weights
    # --------------------------------------------------------
    weight_cols = [
        "attn_instock_w_model",
        "attn_instock_w_last",
        "attn_instock_w_mean4",
        "attn_instock_w_mean13",
    ]

    weight_rows = []
    if all(c in df.columns for c in weight_cols):
        for block, g in df.groupby("horizon_block", observed=True):
            weight_rows.append({
                "block": block,
                "w_model": g["attn_instock_w_model"].mean(),
                "w_last": g["attn_instock_w_last"].mean(),
                "w_mean4": g["attn_instock_w_mean4"].mean(),
                "w_mean13": g["attn_instock_w_mean13"].mean(),
            })

    weights = pd.DataFrame(weight_rows)

    print("\n" + "=" * 100)
    print("D. WHAT DID ATTENTION LEARN? IN_STOCK WEIGHTS BY BLOCK")
    print("=" * 100)
    if len(weights) > 0:
        print(weights.round(4).to_string(index=False))
    else:
        print("No attention weight columns found.")

    # --------------------------------------------------------
    # E. Delta summary
    # --------------------------------------------------------
    raw_overall = overall[overall["method"] == "raw_model"].iloc[0]
    attn_overall = overall[overall["method"] == "anchor_attention"].iloc[0]

    raw_long = block_comp[
        (block_comp["method"] == "raw_model")
        & (block_comp["block"].astype(str) == "long_13_20")
    ].iloc[0]

    attn_long = block_comp[
        (block_comp["method"] == "anchor_attention")
        & (block_comp["block"].astype(str) == "long_13_20")
    ].iloc[0]

    delta = pd.DataFrame([
        {
            "scope": "overall",
            "WAPE_change_attn_minus_raw": attn_overall["WAPE"] - raw_overall["WAPE"],
            "AUC_change_attn_minus_raw": attn_overall["active_AUC"] - raw_overall["active_AUC"],
            "Spearman_change_attn_minus_raw": attn_overall["Spearman"] - raw_overall["Spearman"],
        },
        {
            "scope": "long_13_20",
            "WAPE_change_attn_minus_raw": attn_long["WAPE"] - raw_long["WAPE"],
            "AUC_change_attn_minus_raw": attn_long["active_AUC"] - raw_long["active_AUC"],
            "Spearman_change_attn_minus_raw": attn_long["Spearman"] - raw_long["Spearman"],
        },
    ])

    print("\n" + "=" * 100)
    print("E. ATTENTION IMPROVEMENT SUMMARY")
    print("=" * 100)
    print(delta.round(5).to_string(index=False))

    print("\nInterpretation:")
    print("  WAPE_change < 0 means attention reduces error.")
    print("  AUC_change > 0 means attention improves active/inactive ranking.")
    print("  For your current goal, focus most on long_13_20 AUC_change and WAPE_change.")

    return {
        "overall": overall,
        "block_comparison": block_comp,
        "auc_by_horizon": auc_by_h,
        "attention_weights": weights,
        "delta": delta,
    }


def run_attention_only_focused(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
    history=52,
    horizon=20,
    d_model=64,
    batch_size=64,
    exposure_epochs=60,
    exposure_lr=1e-3,
    exposure_patience=8,
    attn_epochs=80,
    attn_batch_size=1024,
    attn_lr=1e-3,
    attn_patience=10,
    attn_hidden=64,
    attn_dropout=0.10,
    remove_extreme=True,
    extreme_q=0.99,
    suppress_full_print=True,
):
    """
    Clean focused runner.

    This does exactly what you asked:
      1. train direct TCN exposure model
      2. train anchor attention blender
      3. print ONLY attention effect and long-horizon AUC report

    The full long diagnostic tables are suppressed by default.
    """
    import contextlib
    import io

    if suppress_full_print:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            base_result = run_exposure_only(
                data_raw1=data_raw1,
                scot_df=scot_df,
                n_asins=n_asins,
                seed=seed,
                history=history,
                horizon=horizon,
                d_model=d_model,
                batch_size=batch_size,
                epochs=exposure_epochs,
                lr=exposure_lr,
                patience=exposure_patience,
                remove_extreme=remove_extreme,
                extreme_q=extreme_q,
            )

            attn_result = train_and_apply_anchor_attention_blender(
                base_result["forecast_df"],
                targets=("total_dph", "buy_box_dph", "in_stock_dph"),
                epochs=attn_epochs,
                batch_size=attn_batch_size,
                lr=attn_lr,
                patience=attn_patience,
                hidden=attn_hidden,
                dropout=attn_dropout,
                seed=seed,
            )
    else:
        base_result = run_exposure_only(
            data_raw1=data_raw1,
            scot_df=scot_df,
            n_asins=n_asins,
            seed=seed,
            history=history,
            horizon=horizon,
            d_model=d_model,
            batch_size=batch_size,
            epochs=exposure_epochs,
            lr=exposure_lr,
            patience=exposure_patience,
            remove_extreme=remove_extreme,
            extreme_q=extreme_q,
        )

        attn_result = train_and_apply_anchor_attention_blender(
            base_result["forecast_df"],
            targets=("total_dph", "buy_box_dph", "in_stock_dph"),
            epochs=attn_epochs,
            batch_size=attn_batch_size,
            lr=attn_lr,
            patience=attn_patience,
            hidden=attn_hidden,
            dropout=attn_dropout,
            seed=seed,
        )

    attn_df = attn_result["attn_df"]

    key_report = focused_attention_long_auc_report(attn_df)

    exposure_hat_for_demand = make_external_hat_df_from_anchor_attention(
        attn_df,
        include_prob_cols=True,
    )

    print("\n" + "=" * 100)
    print("OUTPUT READY")
    print("=" * 100)
    print("Use result_focus['exposure_hat_for_demand'] for demand model.")
    print("Use result_focus['key_report'] to inspect attention/long-horizon AUC.")

    return {
        "base_result": base_result,
        "attn_result": attn_result,
        "attn_df": attn_df,
        "key_report": key_report,
        "exposure_hat_for_demand": exposure_hat_for_demand,
    }



# ============================================================
# CLEAN DIAGNOSTIC: Best Anchor Attention vs True
# ============================================================

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


def _clean_auc(y_binary, score):
    y_binary = np.asarray(y_binary).astype(int)
    score = np.asarray(score).astype(float)
    mask = np.isfinite(score)
    y_binary = y_binary[mask]
    score = score[mask]
    if len(y_binary) == 0 or len(np.unique(y_binary)) < 2:
        return np.nan
    if roc_auc_score is None:
        return np.nan
    return roc_auc_score(y_binary, score)


def _clean_corr(y, p, method="spearman"):
    y = pd.Series(y).astype(float)
    p = pd.Series(p).astype(float)
    mask = y.notna() & p.notna()
    y = y[mask]
    p = p[mask]
    if len(y) < 3:
        return np.nan
    if y.std() <= 1e-12 or p.std() <= 1e-12:
        return np.nan
    return y.corr(p, method=method)


def _clean_get_best_attn_df(result_obj):
    """
    Get ONLY the best uncalibrated anchor-attention dataframe.

    If result_obj is result_calib:
        use result_calib["result_focus"]["attn_df"] if available
        otherwise result_calib["result_focus"]["exposure_hat_for_demand"]

    It does NOT use calibration.
    It does NOT compare other methods.
    """
    if isinstance(result_obj, dict) and "result_focus" in result_obj:
        rf = result_obj["result_focus"]
        if isinstance(rf, dict) and "attn_df" in rf:
            df = rf["attn_df"].copy()
            source = "result_calib['result_focus']['attn_df']"
        elif isinstance(rf, dict) and "exposure_hat_for_demand" in rf:
            df = rf["exposure_hat_for_demand"].copy()
            source = "result_calib['result_focus']['exposure_hat_for_demand']"
        else:
            raise ValueError("result_calib['result_focus'] has no attn_df or exposure_hat_for_demand.")

    elif isinstance(result_obj, dict) and "attn_df" in result_obj:
        df = result_obj["attn_df"].copy()
        source = "result_focus['attn_df']"

    elif isinstance(result_obj, dict) and "exposure_hat_for_demand" in result_obj:
        df = result_obj["exposure_hat_for_demand"].copy()
        source = "result_focus['exposure_hat_for_demand']"

    else:
        df = result_obj.copy()
        source = "dataframe input"

    print("\n" + "=" * 100)
    print("USING BEST MODEL: UNCALIBRATED ANCHOR ATTENTION ONLY")
    print("=" * 100)
    print("Source:", source)
    print("Rows:", len(df))

    return df


def _clean_pick_cols(df, target):
    """
    Pick true and anchor-attention prediction columns.
    """
    if target == "total_dph":
        true_candidates = ["true_total_dph", "true_future_total_dph", "future_total_dph", "total_dph"]
        pred_candidates = ["attn_total_dph", "pred_total_dph"]

    elif target == "buy_box_dph":
        true_candidates = ["true_buy_box_dph", "true_future_buy_box_dph", "future_buy_box_dph", "buy_box_dph"]
        pred_candidates = ["attn_buy_box_dph", "pred_buy_box_dph"]

    elif target == "in_stock_dph":
        true_candidates = [
            "true_instock_dph", "true_in_stock_dph",
            "true_future_instock", "true_future_instock_dph",
            "future_instock", "in_stock_dph"
        ]
        pred_candidates = [
            "attn_instock_dph", "attn_in_stock_dph",
            "pred_instock_dph", "pred_in_stock_dph"
        ]

    else:
        raise ValueError("target must be total_dph, buy_box_dph, or in_stock_dph")

    true_col = next((c for c in true_candidates if c in df.columns), None)
    pred_col = next((c for c in pred_candidates if c in df.columns), None)

    if true_col is None:
        raise ValueError(f"Cannot find true column for {target}. Available columns: {df.columns.tolist()}")
    if pred_col is None:
        raise ValueError(f"Cannot find attention prediction column for {target}. Available columns: {df.columns.tolist()}")

    return true_col, pred_col


def _clean_metrics(g, true_col, pred_col, active_threshold=0.0):
    y = pd.to_numeric(g[true_col], errors="coerce").fillna(0.0).clip(lower=0.0).values
    p = pd.to_numeric(g[pred_col], errors="coerce").fillna(0.0).clip(lower=0.0).values
    eps = 1e-8

    return {
        "n": len(g),
        "true_mean": y.mean(),
        "pred_mean": p.mean(),
        "ratio": p.mean() / (y.mean() + eps),
        "WAPE": np.abs(p - y).sum() / (np.abs(y).sum() + eps),
        "underbias": np.maximum(y - p, 0.0).sum() / (np.abs(y).sum() + eps),
        "overbias": np.maximum(p - y, 0.0).sum() / (np.abs(y).sum() + eps),
        "log_MAE": np.abs(np.log1p(p) - np.log1p(y)).mean(),
        "active_rate": (y > active_threshold).mean(),
        "active_AUC": _clean_auc(y > active_threshold, p),
        "Spearman": _clean_corr(y, p, "spearman"),
        "Pearson": _clean_corr(y, p, "pearson"),
    }


def diagnose_best_anchor_attention_vs_true(
    result_obj,
    targets=("total_dph", "buy_box_dph", "in_stock_dph"),
    horizon_col="horizon",
    asin_col="asin",
    active_threshold=0.0,
):
    """
    Clean diagnostic for the best exposure model only:
        Direct TCN + Anchor Attention
        No calibration
        No raw model comparison
        No last/mean baseline comparison

    Outputs exactly three diagnostic dimensions:
      1. length / horizon h1-h20
      2. active
      3. magnitude

    Plus compact ASIN 20-week summary.
    """
    df = _clean_get_best_attn_df(result_obj)

    all_results = {}
    summary_rows = []

    for target in targets:
        true_col, pred_col = _clean_pick_cols(df, target)

        print("\n" + "#" * 100)
        print(f"TARGET: {target}")
        print("#" * 100)
        print("true_col:", true_col)
        print("pred_col:", pred_col)

        d = df.copy()
        d["_true"] = pd.to_numeric(d[true_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        d["_pred"] = pd.to_numeric(d[pred_col], errors="coerce").fillna(0.0).clip(lower=0.0)

        overall = pd.DataFrame([_clean_metrics(d, "_true", "_pred", active_threshold)])
        overall.insert(0, "target", target)

        print("\nOVERALL")
        print(overall.round(5).to_string(index=False))

        # 1. Length / horizon h1-h20
        length_table = None
        active_table = None
        mag_table = None

        if horizon_col in d.columns:
            rows = []
            for h, g in d.groupby(horizon_col):
                row = _clean_metrics(g, "_true", "_pred", active_threshold)
                row["horizon"] = h
                rows.append(row)

            by_h = pd.DataFrame(rows).sort_values("horizon")

            length_table = by_h[[
                "horizon", "n", "true_mean", "pred_mean", "ratio", "WAPE"
            ]].copy()

            active_table = by_h[[
                "horizon", "active_rate", "active_AUC", "Spearman", "Pearson"
            ]].copy()

            mag_table = by_h[[
                "horizon", "true_mean", "pred_mean", "ratio",
                "WAPE", "underbias", "overbias", "log_MAE"
            ]].copy()

            print("\n1. LENGTH / HORIZON h1-h20")
            print(length_table.round(5).to_string(index=False))

            print("\n2. ACTIVE BY HORIZON")
            print(active_table.round(5).to_string(index=False))

            print("\n3. MAGNITUDE BY HORIZON")
            print(mag_table.round(5).to_string(index=False))

        else:
            print(f"\nNo horizon column found: {horizon_col}")

        # True magnitude groups: low/mid/high
        d["_true_group"] = pd.qcut(
            d["_true"].rank(method="first"),
            q=3,
            labels=["low_true", "mid_true", "high_true"],
        )

        group_rows = []
        for grp, g in d.groupby("_true_group", observed=True):
            row = _clean_metrics(g, "_true", "_pred", active_threshold)
            row["group"] = str(grp)
            group_rows.append(row)

        group_table = pd.DataFrame(group_rows)[[
            "group", "n", "true_mean", "pred_mean", "ratio",
            "WAPE", "underbias", "overbias", "active_AUC", "Spearman"
        ]]

        print("\nMAGNITUDE GROUP: LOW / MID / HIGH TRUE")
        print(group_table.round(5).to_string(index=False))

        # ASIN-level 20-week sum
        asin_table = None
        if asin_col in d.columns:
            eps = 1e-8
            asin_sum = (
                d.groupby(asin_col)
                .agg(
                    true_sum=("_true", "sum"),
                    pred_sum=("_pred", "sum"),
                )
                .reset_index()
            )
            asin_sum["abs_err"] = (asin_sum["pred_sum"] - asin_sum["true_sum"]).abs()
            asin_sum["ratio"] = asin_sum["pred_sum"] / (asin_sum["true_sum"] + eps)
            asin_sum["asin_wape"] = asin_sum["abs_err"] / (asin_sum["true_sum"] + eps)

            asin_table = pd.DataFrame([{
                "target": target,
                "n_asins": asin_sum[asin_col].nunique(),
                "asin_sum_Spearman": _clean_corr(asin_sum["true_sum"], asin_sum["pred_sum"], "spearman"),
                "asin_sum_Pearson": _clean_corr(asin_sum["true_sum"], asin_sum["pred_sum"], "pearson"),
                "median_asin_wape": asin_sum["asin_wape"].median(),
                "p75_asin_wape": asin_sum["asin_wape"].quantile(0.75),
                "p90_asin_wape": asin_sum["asin_wape"].quantile(0.90),
                "median_sum_ratio": asin_sum["ratio"].median(),
            }])

            print("\nASIN 20-WEEK SUM")
            print(asin_table.round(5).to_string(index=False))

        o = overall.iloc[0]
        summary_row = {
            "target": target,
            "ratio": o["ratio"],
            "WAPE": o["WAPE"],
            "underbias": o["underbias"],
            "overbias": o["overbias"],
            "active_AUC": o["active_AUC"],
            "Spearman": o["Spearman"],
        }

        if asin_table is not None:
            a = asin_table.iloc[0]
            summary_row["asin_sum_Spearman"] = a["asin_sum_Spearman"]
            summary_row["p90_asin_wape"] = a["p90_asin_wape"]

        summary_rows.append(summary_row)

        all_results[target] = {
            "overall": overall,
            "length_by_horizon": length_table,
            "active_by_horizon": active_table,
            "magnitude_by_horizon": mag_table,
            "group_table": group_table,
            "asin_table": asin_table,
            "scored_df": d,
        }

    summary = pd.DataFrame(summary_rows)

    print("\n" + "#" * 100)
    print("FINAL COMPACT SUMMARY")
    print("#" * 100)
    print(summary.round(5).to_string(index=False))

    all_results["summary"] = summary
    return all_results


# ============================================================
# Only usage you need
# ============================================================
#
# exposure_diag = diagnose_best_anchor_attention_vs_true(result_calib)
#
# If only in_stock:
# exposure_diag_instock = diagnose_best_anchor_attention_vs_true(
#     result_calib,
#     targets=("in_stock_dph",),
# )
#



# ============================================================
# 17. Category Cross Features: CURRENT static values only
# ============================================================
# User-requested version:
#   Use ONLY:
#       gl_product_group
#       category_code
#       ind_top10_brand
#
#   Use current/static values directly.
#   No lag group summaries.
#   No total_dph / buy_box_dph / in_stock_dph group aggregates.
#   No future exposure leakage.
#
# Meaning:
#   These are product/category identity features.
#   They are treated as known current ASIN attributes and used to predict future exposure.
# ============================================================


def _pick_category_cross_cols(df):
    """
    Use only the three requested current/static category features.
    """
    requested_cols = [
        "gl_product_group",
        "category_code",
        "ind_top10_brand",
    ]
    return [c for c in requested_cols if c in df.columns]


def _encode_category_cross_features(df, cat_cols):
    """
    Encode only current/static category features.

    For each column c, create:
      cross_cat__c__code
      cross_cat__c__freq

    These are not lagged because they are static/current ASIN attributes.
    """
    df = df.copy()
    out_cols = []

    for c in cat_cols:
        s = df[c].astype(str).fillna("MISSING")

        code_col = f"cross_cat__{c}__code"
        freq_col = f"cross_cat__{c}__freq"

        codes, uniques = pd.factorize(s)
        denom = max(len(uniques) - 1, 1)

        df[code_col] = codes.astype(float) / denom
        df[freq_col] = s.map(s.value_counts(normalize=True)).astype(float).fillna(0.0)

        out_cols.extend([code_col, freq_col])

    return df, out_cols


def _build_cross_attention_context_columns(data_raw):
    """
    Add ONLY current/static category features.

    No lag features.
    No group historical exposure features.
    """
    df = data_raw.copy()

    cat_cols = _pick_category_cross_cols(df)
    df, cross_cols = _encode_category_cross_features(df, cat_cols)

    print("\n" + "=" * 100)
    print("CURRENT CATEGORY FEATURES FOR EXPOSURE MODEL")
    print("=" * 100)
    print("Requested raw category columns:")
    print(["gl_product_group", "category_code", "ind_top10_brand"])
    print("\nFound raw category columns:")
    print(cat_cols)
    print("\nEncoded feature columns added:")
    print(cross_cols)
    print("\nNo lag group summaries are used.")

    return df, cross_cols


def add_current_category_features_to_exposure_data(data_raw):
    """
    Public helper.

    Usage:
      data_raw_cat, cat_feature_cols = add_current_category_features_to_exposure_data(data_raw1)
    """
    return _build_cross_attention_context_columns(data_raw)


def diagnose_current_category_feature_quality(data_raw):
    """
    Quick check for the three requested category features.
    """
    df, cross_cols = _build_cross_attention_context_columns(data_raw)

    rows = []
    for raw_col in ["gl_product_group", "category_code", "ind_top10_brand"]:
        rows.append({
            "raw_col": raw_col,
            "exists": raw_col in data_raw.columns,
            "n_unique": data_raw[raw_col].nunique(dropna=False) if raw_col in data_raw.columns else np.nan,
            "missing_rate": data_raw[raw_col].isna().mean() if raw_col in data_raw.columns else np.nan,
        })

    raw_summary = pd.DataFrame(rows)

    enc_rows = []
    for c in cross_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        enc_rows.append({
            "encoded_col": c,
            "missing_rate": s.isna().mean(),
            "mean": s.fillna(0.0).mean(),
            "std": s.fillna(0.0).std(),
            "min": s.fillna(0.0).min(),
            "max": s.fillna(0.0).max(),
        })

    enc_summary = pd.DataFrame(enc_rows)

    print("\n" + "=" * 100)
    print("RAW CATEGORY FEATURE QUALITY")
    print("=" * 100)
    print(raw_summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("ENCODED CATEGORY FEATURE QUALITY")
    print("=" * 100)
    if len(enc_summary) > 0:
        print(enc_summary.round(5).to_string(index=False))
    else:
        print("No encoded category features were created.")

    return {
        "raw_summary": raw_summary,
        "encoded_summary": enc_summary,
        "data_with_category_features": df,
        "category_feature_cols": cross_cols,
    }


def run_exposure_anchor_attention_with_current_category_features(
    data_raw1,
    scot_df=None,
    base_runner_kwargs=None,
):
    """
    Compare current best Anchor Attention exposure model
    with ONLY current/static category features added:

      gl_product_group
      category_code
      ind_top10_brand

    No lag features.
    No group aggregate exposure.
    No calibration.

    This runs your existing best exposure pipeline on data_raw1 augmented with
    encoded current category features.

    After this, run:
      cross_diag = diagnose_best_anchor_attention_vs_true(result_cat)
    """
    if base_runner_kwargs is None:
        base_runner_kwargs = {}

    data_aug, cat_feature_cols = _build_cross_attention_context_columns(data_raw1)

    print("\n" + "=" * 100)
    print("RUNNING BEST ANCHOR ATTENTION WITH CURRENT CATEGORY FEATURES ONLY")
    print("=" * 100)
    print("Added category feature count:", len(cat_feature_cols))
    print("Features:")
    print(cat_feature_cols)

    if "run_attention_focused_with_calibration" in globals():
        result = run_attention_focused_with_calibration(
            data_raw1=data_aug,
            scot_df=scot_df,
            **base_runner_kwargs,
        )
    elif "run_attention_only_focused" in globals():
        result = run_attention_only_focused(
            data_raw1=data_aug,
            scot_df=scot_df,
            **base_runner_kwargs,
        )
    else:
        raise NameError(
            "Cannot find run_attention_focused_with_calibration or run_attention_only_focused."
        )

    result["current_category_feature_cols"] = cat_feature_cols
    result["data_raw_with_current_category_features"] = data_aug

    print("\n" + "=" * 100)
    print("CURRENT CATEGORY FEATURE RUN FINISHED")
    print("=" * 100)
    print("Now compare:")
    print("  base: diagnose_best_anchor_attention_vs_true(result_calib)")
    print("  cat : diagnose_best_anchor_attention_vs_true(result_cat)")

    return result


# Backward-compatible alias, so old usage still works but now only uses the three current category features.
def run_exposure_anchor_attention_with_category_cross_attention(
    data_raw1,
    scot_df=None,
    base_runner_kwargs=None,
):
    """
    Alias to current-category-only version.
    """
    return run_exposure_anchor_attention_with_current_category_features(
        data_raw1=data_raw1,
        scot_df=scot_df,
        base_runner_kwargs=base_runner_kwargs,
    )



# ============================================================
# 18. Clean comparison: Base best vs + current category cross features
# ============================================================

def run_base_best_anchor_attention_exposure(
    data_raw1,
    scot_df=None,
    n_asins=5000,
    seed=42,
    history=52,
    horizon=20,
    d_model=64,
    batch_size=64,
    exposure_epochs=60,
    exposure_lr=1e-3,
    exposure_patience=8,
    suppress_full_print=True,
):
    """
    Version A:
      Original best exposure model:
        Direct TCN + Anchor Attention

    Keeps:
      last / mean4 / mean13 anchors

    Does NOT add:
      category cross features
      calibration
    """
    print("\n" + "=" * 100)
    print("VERSION A: BASE BEST ANCHOR ATTENTION")
    print("=" * 100)
    print("Keeps last / mean4 / mean13 anchors.")
    print("No current category cross features.")

    if "run_attention_focused_with_calibration" in globals():
        result_base = run_attention_focused_with_calibration(
            data_raw1=data_raw1,
            scot_df=scot_df,
            n_asins=n_asins,
            seed=seed,
            history=history,
            horizon=horizon,
            d_model=d_model,
            batch_size=batch_size,
            exposure_epochs=exposure_epochs,
            exposure_lr=exposure_lr,
            exposure_patience=exposure_patience,
            suppress_full_print=suppress_full_print,
        )
    elif "run_attention_only_focused" in globals():
        result_base = run_attention_only_focused(
            data_raw1=data_raw1,
            scot_df=scot_df,
            n_asins=n_asins,
            seed=seed,
            history=history,
            horizon=horizon,
            d_model=d_model,
            batch_size=batch_size,
            exposure_epochs=exposure_epochs,
            exposure_lr=exposure_lr,
            exposure_patience=exposure_patience,
            suppress_full_print=suppress_full_print,
        )
    else:
        raise NameError("Cannot find run_attention_focused_with_calibration or run_attention_only_focused.")

    return result_base


def run_category_cross_anchor_attention_exposure(
    data_raw1,
    scot_df=None,
    n_asins=5000,
    seed=42,
    history=52,
    horizon=20,
    d_model=64,
    batch_size=64,
    exposure_epochs=60,
    exposure_lr=1e-3,
    exposure_patience=8,
    suppress_full_print=True,
):
    """
    Version B:
      Base best exposure model + current/static category cross features.

    Keeps:
      last / mean4 / mean13 anchors

    Adds ONLY:
      gl_product_group
      category_code
      ind_top10_brand

    Does NOT add:
      lag group summaries
      future exposure leakage
      calibration
    """
    print("\n" + "=" * 100)
    print("VERSION B: BASE BEST + CURRENT CATEGORY CROSS FEATURES")
    print("=" * 100)
    print("Keeps last / mean4 / mean13 anchors.")
    print("Adds only: gl_product_group, category_code, ind_top10_brand.")
    print("No lag group summaries.")

    result_cat = run_exposure_anchor_attention_with_current_category_features(
        data_raw1=data_raw1,
        scot_df=scot_df,
        base_runner_kwargs={
            "n_asins": n_asins,
            "seed": seed,
            "history": history,
            "horizon": horizon,
            "d_model": d_model,
            "batch_size": batch_size,
            "exposure_epochs": exposure_epochs,
            "exposure_lr": exposure_lr,
            "exposure_patience": exposure_patience,
            "suppress_full_print": suppress_full_print,
        },
    )

    return result_cat


def compare_base_vs_category_cross_exposure(
    data_raw1,
    scot_df=None,
    n_asins=5000,
    seed=42,
    history=52,
    horizon=20,
    d_model=64,
    batch_size=64,
    exposure_epochs=60,
    exposure_lr=1e-3,
    exposure_patience=8,
    suppress_full_print=True,
):
    """
    Run and diagnose the two versions you want:

      A. Original best:
           Direct TCN + Anchor Attention
           with last / mean4 / mean13 anchors

      B. + Current category cross features:
           A + gl_product_group / category_code / ind_top10_brand

    Output:
      result_compare["base_result"]
      result_compare["cat_result"]
      result_compare["base_diag"]
      result_compare["cat_diag"]
      result_compare["summary_compare"]
    """
    result_base = run_base_best_anchor_attention_exposure(
        data_raw1=data_raw1,
        scot_df=scot_df,
        n_asins=n_asins,
        seed=seed,
        history=history,
        horizon=horizon,
        d_model=d_model,
        batch_size=batch_size,
        exposure_epochs=exposure_epochs,
        exposure_lr=exposure_lr,
        exposure_patience=exposure_patience,
        suppress_full_print=suppress_full_print,
    )

    base_diag = diagnose_best_anchor_attention_vs_true(result_base)

    result_cat = run_category_cross_anchor_attention_exposure(
        data_raw1=data_raw1,
        scot_df=scot_df,
        n_asins=n_asins,
        seed=seed,
        history=history,
        horizon=horizon,
        d_model=d_model,
        batch_size=batch_size,
        exposure_epochs=exposure_epochs,
        exposure_lr=exposure_lr,
        exposure_patience=exposure_patience,
        suppress_full_print=suppress_full_print,
    )

    cat_diag = diagnose_best_anchor_attention_vs_true(result_cat)

    # Compact comparison
    base_summary = base_diag["summary"].copy()
    cat_summary = cat_diag["summary"].copy()

    base_summary = base_summary.add_prefix("base_")
    cat_summary = cat_summary.add_prefix("cat_")

    compare = pd.concat([base_summary, cat_summary], axis=1)

    # Restore target column
    if "base_target" in compare.columns:
        compare.insert(0, "target", compare["base_target"])

    # Add delta columns. Negative WAPE/underbias is improvement.
    metric_pairs = [
        ("WAPE", "lower_better"),
        ("underbias", "lower_better"),
        ("overbias", "lower_better"),
        ("active_AUC", "higher_better"),
        ("Spearman", "higher_better"),
    ]

    for m, direction in metric_pairs:
        b = f"base_{m}"
        c = f"cat_{m}"
        if b in compare.columns and c in compare.columns:
            compare[f"delta_{m}_cat_minus_base"] = compare[c] - compare[b]

    if "base_p90_asin_wape" in compare.columns and "cat_p90_asin_wape" in compare.columns:
        compare["delta_p90_asin_wape_cat_minus_base"] = (
            compare["cat_p90_asin_wape"] - compare["base_p90_asin_wape"]
        )

    print("\n" + "#" * 100)
    print("BASE vs CATEGORY CROSS COMPACT COMPARISON")
    print("#" * 100)
    print(compare.round(5).to_string(index=False))

    print("\nHow to read:")
    print("  delta_WAPE_cat_minus_base < 0 means category version improved WAPE.")
    print("  delta_underbias_cat_minus_base < 0 means category version reduced underprediction.")
    print("  delta_active_AUC_cat_minus_base > 0 means category version improved active ranking.")
    print("  For long horizon, compare h13-h20 tables inside base_diag and cat_diag.")

    return {
        "base_result": result_base,
        "cat_result": result_cat,
        "base_diag": base_diag,
        "cat_diag": cat_diag,
        "summary_compare": compare,
    }


# ============================================================
# Recommended two-version comparison usage
# ============================================================
#
# result_compare = compare_base_vs_category_cross_exposure(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     n_asins=5000,
#     seed=42,
#     history=52,
#     horizon=20,
#     d_model=64,
#     batch_size=64,
#     exposure_epochs=60,
#     exposure_lr=1e-3,
#     exposure_patience=8,
#     suppress_full_print=True,
# )
#
