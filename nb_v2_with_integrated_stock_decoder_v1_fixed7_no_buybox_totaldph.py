"""
TCN + Sparse Attention + ENN with Negative Binomial likelihood.

This version runs on the high-sparse group and includes:
- leak-safe rolling positive features
- soft sparse attention mask
- dilation sequence [1, 2, 4, 8, 13, 26, 52]
- early stopping
- z regularization
- encoder diagnostics
- final WAPE summary
- history uses raw in_stock_dph; future context excludes in_stock_dph
- future context includes distance-to-holiday scalar features
- stock decoder uses extra product/popularity/promo/package features
- stock decoder extra future context excludes true future total_dph and buy_box_dph
- safe historical total_dph/buy_box_dph proxy features are repeated across horizon
- total amount diagnostics: sum(fbi_demand * raw our_price)
- total size diagnostics: sum(fbi_demand * pkg_height * pkg_length * pkg_width)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, r2_score

torch.manual_seed(42)
np.random.seed(42)


# =====================================================
# 0. Sampling
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
    df["order_week"] = pd.to_datetime(df["order_week"])
    scot["asin"] = scot["asin"].astype(str)

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
    print("Sample ASINs missing in SCOT:", len(sample_asin_set - scot_asin_set))

    data_small = df[df["asin"].isin(intersect_asins)].copy()
    sample_asin_df = pd.DataFrame({"asin": list(sample_asins)})
    intersect_asin_df = pd.DataFrame({"asin": intersect_asins})

    print("Data rows after intersection:", len(data_small))
    print("Data ASINs after intersection:", data_small["asin"].nunique())

    return data_small, sample_asin_df, intersect_asin_df


def add_zero_rate_group(data_raw, zero_thresholds=(0.4, 0.7)):
    df = data_raw.copy()
    df["fbi_demand"] = pd.to_numeric(df["fbi_demand"], errors="coerce").fillna(0).clip(lower=0)
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
        if z < low: return "low_sparse"
        elif z < high: return "mid_sparse"
        else: return "high_sparse"
    asin_stats["zero_group"] = asin_stats["zero_rate"].apply(assign_group)
    df = df.merge(asin_stats[["asin", "zero_rate", "zero_group"]], on="asin", how="left")
    print("\nASIN counts by zero-rate group:")
    print(asin_stats.groupby("zero_group")["asin"].nunique().reset_index(name="n_asins"))
    return df, asin_stats


# =====================================================
# 1. Data loading
# =====================================================


def _infer_pkg_dimension_cols(df):
    """
    Infer package height, length, and width columns for package-volume diagnostics.
    Diagnostic only; not used as model input.
    """
    lower_map = {c.lower(): c for c in df.columns}

    candidates = {
        "height": [
            "pkg_height", "package_height", "pkg_h", "height",
            "item_height", "unit_height"
        ],
        "length": [
            "pkg_length", "package_length", "pkg_l", "length",
            "item_length", "unit_length"
        ],
        "width": [
            "pkg_width", "package_width", "pkg_w", "width",
            "item_width", "unit_width"
        ],
    }

    out = {}

    for dim_name, names in candidates.items():
        out[dim_name] = None
        for name in names:
            if name in lower_map:
                out[dim_name] = lower_map[name]
                break

    return out




def _get_1d_col(df, col):
    """
    Return one 1-D Series even if df has duplicate column names.
    """
    x = df[col]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x


def _select_stock_decoder_extra_cols(data_raw):
    """
    Select additional features to help the stock decoder predict future in_stock_dph_hat.

    These are NOT true future in_stock_dph. They are product / popularity / price / promo
    / package features that can help predict future exposure.

    We keep a conservative list to avoid leakage-prone realized future outcomes.
    """
    candidate_cols = [
        # Product/category/static identity proxies
        "gl_product_group",
        "category_code",
        "brand_class",
        "sort_type",
        "variation",
        "ind_new_asin",
        "ind_amxl_hb",
        "hbt",
        "ind_target_audience",
        "ind_top10_brand",
        "ind_top10_review_brand",

        # Review / popularity proxies.
        # NOTE: total_dph and buy_box_dph are intentionally excluded here
        # because future realized traffic / buy-box signals may cause leakage.
        "cust_avg_active_review_rating",
        "customer_active_review_count",
        "customer_average_review_rating",
        "customer_review_count",
        "glance_view_band_cat",
        "hb_rank",
        "hb_score",
        "facebook_fan_count",
        "instagram_fan_count",
        "twitter_follower_count",
        "youtube_subscriber_count",

        # Price / promotion
        "list_price",
        "price_bands",
        "ind_promotion",
        "promotion_amount",
        "promotion_ratio",
        "promotion_pricing_amount",
        "promotion_type",
        "pricing_type",
        "asin_promo_start_week",
        "asin_promo_end_week",
        "asin_promo_wordcount",

        # Package / AMXL size
        "pkg_height",
        "pkg_length",
        "pkg_width",
        "pkg_weight",

        # Calendar-ish columns
        "order_month",
        "order_year",
        "week_index",
        "ind_prime_week",
    ]

    # Avoid realized target / future outcome columns.
    exclude_cols = {
        "fbi_demand",
        "order_units",
        "scot_oos",
        "in_stock_dph",
        "asin",
        "order_week",
    }

    cols = [
        c for c in candidate_cols
        if c in data_raw.columns and c not in exclude_cols
    ]

    return cols


def _encode_stock_decoder_extra_features(df, extra_cols):
    """
    Convert extra stock-decoder features to numeric features.

    Object/categorical columns are ordinal-encoded by pandas.factorize.
    This keeps the implementation lightweight and avoids requiring sklearn encoders.
    """
    out_cols = []

    for c in extra_cols:
        new_c = f"stock_extra__{c}"

        if c not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            val = pd.to_numeric(_get_1d_col(df, c), errors="coerce").fillna(0.0)

            # Conservative transforms by feature type.
            cl = c.lower()
            if (
                "count" in cl or "dph" in cl or "price" in cl
                or "amount" in cl or "rank" in cl or "score" in cl
                or "height" in cl or "length" in cl or "width" in cl
                or "weight" in cl or "wordcount" in cl
            ):
                val = np.log1p(val.clip(lower=0))

            # Scale robustly to avoid huge values.
            std = float(val.std()) if float(val.std()) > 1e-8 else 1.0
            mean = float(val.mean())
            df[new_c] = ((val - mean) / std).clip(-5, 5)

        else:
            codes, uniques = pd.factorize(_get_1d_col(df, c).astype(str).fillna("MISSING"))
            # normalize category code to roughly [0,1]
            denom = max(len(uniques) - 1, 1)
            df[new_c] = codes.astype(float) / denom

        out_cols.append(new_c)

    return df, out_cols



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
    """
    FIX: arr[lo:i] not arr[lo:i+1]
    Excludes current timestep to prevent data leakage.
    """
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window)
        vals = arr[lo:i]          # ← FIX: exclude current step
        vals = vals[vals > 0]
        out[i] = vals.mean() if len(vals) > 0 else 0.0
    return out


def _rolling_positive_quantile(arr, window, q):
    """
    FIX: arr[lo:i] not arr[lo:i+1]
    Excludes current timestep to prevent data leakage.
    """
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window)
        vals = arr[lo:i]          # ← FIX: exclude current step
        vals = vals[vals > 0]
        out[i] = np.quantile(vals, q) if len(vals) > 0 else 0.0
    return out


def _rolling_max_lag(arr, window):
    """Lag-safe rolling max excluding current step."""
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window)
        vals = arr[lo:i]
        out[i] = vals.max() if len(vals) > 0 else 0.0
    return out


def _zero_streak(active):
    out = np.zeros(len(active), dtype=np.float32)
    cur = 0
    for i, a in enumerate(active):
        if a > 0: cur = 0
        else: cur += 1
        out[i] = cur
    return out


def load_real_data(data_raw):
    """
    25 history features, all leak-free.
    Feature index map:
      0  log1p(demand)
      1  active indicator
      2  distance since last active / 52
      3  sin(2π t/52)
      4  cos(2π t/52)
      5  promo_t
      6  sin(2π t/13)
      7  cos(2π t/13)
      8  hist_nonzero_mean_52_log   ← lag-fixed
      9  hist_nonzero_p75_52_log    ← lag-fixed
      10 recent_peak_13_log         ← lag-fixed
      11 in_stock_dph_lag_log
      12 oos
      13 active_rate_4
      14 active_rate_13
      15 oos_rate_4
      16 oos_rate_13
      17 instock_mean_4_log
      18 instock_mean_13_log
      19 zero_streak_scaled
      20 price_log
      21 positive_mean_4_log        ← lag-fixed
      22 positive_mean_13_log       ← lag-fixed
      23 positive_max_13_log        ← lag-fixed
      24 positive_std_13
    """
    holiday_cols = [c for c in data_raw.columns if c.startswith("holiday_indicator_")]
    distance_cols = [c for c in data_raw.columns if c.startswith("distance_")]
    stock_extra_raw_cols = _select_stock_decoder_extra_cols(data_raw)
    pkg_cols = _infer_pkg_dimension_cols(data_raw)
    # Future-known context features. Exclude in_stock_dph because future traffic is unknown.
    # Future-known context features. Exclude in_stock_dph because future traffic is unknown.
    # Include holiday indicators and distance-to-holiday scalar features.
    context_cols = ["our_price"] + holiday_cols + distance_cols

    base_cols = ["asin", "order_week", "fbi_demand", "scot_oos"]

    # Keep in_stock_dph for history encoder only.
    # It is intentionally excluded from future_context.
    # Keep DPH variables for history-only safe proxy features.
    # They are not used as raw future context.
    history_only_cols = ["in_stock_dph", "total_dph", "buy_box_dph"]

    extra_diag_cols = [c for c in pkg_cols.values() if c is not None]

    keep_cols = [
        c for c in base_cols + context_cols + history_only_cols + extra_diag_cols + stock_extra_raw_cols
        if c in data_raw.columns
    ]

    # Remove duplicate column names. Duplicates can happen because package columns
    # are used both for total_size diagnostics and stock-decoder extra features.
    keep_cols = list(dict.fromkeys(keep_cols))

    df = data_raw[keep_cols].copy()

    # Encode additional product / popularity / promo / size features for stock decoder.
    df, stock_extra_cols = _encode_stock_decoder_extra_features(df, stock_extra_raw_cols)

    # Add encoded stock-extra columns to future_context.
    # These features help the stock decoder predict future in_stock_dph_hat.
    context_cols = context_cols + stock_extra_cols

    # Forecast-origin-safe historical DPH proxy features.
    # These columns are placeholders here and are filled inside DemandDataset
    # using only history up to each forecast origin.
    dph_proxy_cols = [
        "hist_total_dph_last_log",
        "hist_total_dph_mean4_log",
        "hist_total_dph_mean13_log",
        "hist_buy_box_dph_last_log",
        "hist_buy_box_dph_mean4_log",
        "hist_buy_box_dph_mean13_log",
    ]
    for c in dph_proxy_cols:
        df[c] = 0.0

    context_cols = context_cols + dph_proxy_cols
    df = df.rename(columns={"asin":"ASIN","order_week":"Week","fbi_demand":"Demand","scot_oos":"OOS"})

    h_col = pkg_cols.get("height")
    l_col = pkg_cols.get("length")
    w_col = pkg_cols.get("width")

    if h_col is not None and l_col is not None and w_col is not None:
        pkg_h = pd.to_numeric(_get_1d_col(df, h_col), errors="coerce").fillna(0).clip(lower=0)
        pkg_l = pd.to_numeric(_get_1d_col(df, l_col), errors="coerce").fillna(0).clip(lower=0)
        pkg_w = pd.to_numeric(_get_1d_col(df, w_col), errors="coerce").fillna(0).clip(lower=0)
        df["pkg_volume_raw"] = pkg_h * pkg_l * pkg_w
    else:
        df["pkg_volume_raw"] = np.nan

    df["Week"] = pd.to_datetime(df["Week"])
    df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce").fillna(0).clip(lower=0)
    df["OOS"] = pd.to_numeric(df["OOS"], errors="coerce").fillna(0)
    for c in context_cols:
        df = _safe_numeric(df, c, default=0.0)

    # Keep raw price for amount diagnostics, then use log price for model context.
    df["our_price_raw"] = df["our_price"].clip(lower=0)
    df["our_price"] = np.log1p(df["our_price_raw"])

    # Use historical in_stock_dph directly in the encoder; no lag shift.
    # Future in_stock_dph is not used in future_context.
    if "in_stock_dph" in df.columns:
        df["in_stock_dph"] = pd.to_numeric(df["in_stock_dph"], errors="coerce").fillna(0.0)
        df["in_stock_dph"] = df["in_stock_dph"].clip(lower=0)
    else:
        df["in_stock_dph"] = 0.0

    # Historical total_dph / buy_box_dph are used only as forecast-origin-safe summaries.
    for c in ["total_dph", "buy_box_dph"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0)
        else:
            df[c] = 0.0
    for c in holiday_cols:
        df[c] = df[c].clip(lower=0, upper=1)

    # Distance-to-holiday features are future-known scalar calendar features.
    # Keep direction if raw values are signed: negative = before holiday, positive = after holiday.
    for c in distance_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df[c] = df[c].clip(lower=-12, upper=12) / 12.0

    df = df.sort_values(["ASIN", "Week"]).reset_index(drop=True)

    if len(holiday_cols) > 0:
        holiday_window = np.zeros(len(df), dtype=np.float32)
        for c in holiday_cols:
            cur = df[c].values.astype(float)
            prev_window = np.roll(cur, -1); prev_window[-1] = 0
            holiday_window = np.maximum(holiday_window, np.maximum(cur, prev_window))
        df["promo_t"] = holiday_window
    else:
        df["promo_t"] = 0.0

    df["t"] = ((df["Week"] - df["Week"].min()).dt.days // 7).astype(int)

    data = {}
    for asin, group in df.groupby("ASIN"):
        group = group.reset_index(drop=True)
        demand = group["Demand"].values.astype(float)
        oos    = group["OOS"].values.astype(float)
        weeks  = group["Week"].values
        t      = group["t"].values
        T      = len(demand)

        v_t = np.log1p(demand)
        b_t = (demand > 0).astype(float)

        d_t = np.zeros(T)
        last = -1
        for i in range(T):
            if b_t[i] > 0: last = i
            d_t[i] = (i - last) / 52.0 if last >= 0 else 1.0

        in_stock_lag = group["in_stock_dph"].values.astype(float)
        instock_raw  = group["in_stock_dph"].values.astype(float)
        price_log    = group["our_price"].values.astype(float)
        price_raw    = group["our_price_raw"].values.astype(float)
        pkg_volume_raw = group["pkg_volume_raw"].values.astype(float)
        total_dph_raw = group["total_dph"].values.astype(float)
        buy_box_dph_raw = group["buy_box_dph"].values.astype(float)

        # All rolling features now exclude current step (leak-free)
        hist_nonzero_mean_52 = _rolling_positive_mean(demand, 52)
        hist_nonzero_p75_52  = _rolling_positive_quantile(demand, 52, 0.75)
        recent_peak_13       = _rolling_max_lag(demand, 13)

        active_rate_4   = _rolling_mean(b_t, 4)
        active_rate_13  = _rolling_mean(b_t, 13)
        oos_rate_4      = _rolling_mean(oos, 4)
        oos_rate_13     = _rolling_mean(oos, 13)
        instock_mean_4  = _rolling_mean(in_stock_lag, 4)
        instock_mean_13 = _rolling_mean(in_stock_lag, 13)
        zero_streak     = _zero_streak(b_t) / 52.0

        positive_mean_4  = _rolling_positive_mean(demand, 4)
        positive_mean_13 = _rolling_positive_mean(demand, 13)
        positive_max_13  = _rolling_max_lag(demand, 13)
        positive_std_13  = _rolling_std(np.log1p(demand), 13)

        features = np.stack([
            v_t,
            b_t,
            d_t,
            np.sin(2 * np.pi * t / 52),
            np.cos(2 * np.pi * t / 52),
            group["promo_t"].values.astype(float),
            np.sin(2 * np.pi * t / 13),
            np.cos(2 * np.pi * t / 13),
            np.log1p(hist_nonzero_mean_52),   # 8
            np.log1p(hist_nonzero_p75_52),    # 9
            np.log1p(recent_peak_13),         # 10
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
        ], axis=1).astype(np.float32)

        future_context = group[context_cols].values.astype(np.float32)


        data[asin] = {
            "features": features,
            "future_context": future_context,
            "demand": demand.astype(np.float32),
            "week": weeks,
            "oos": oos.astype(np.float32),
            "price_raw": price_raw.astype(np.float32),
            "pkg_volume_raw": pkg_volume_raw.astype(np.float32),
            "instock_raw": instock_raw.astype(np.float32),
            "total_dph_raw": total_dph_raw.astype(np.float32),
            "buy_box_dph_raw": buy_box_dph_raw.astype(np.float32),
            "dph_proxy_context_idx": {
                c: context_cols.index(c) for c in dph_proxy_cols if c in context_cols
            },
        }

    print("History encoder dim: 25")
    print(f"Package dimension columns for total_size: {pkg_cols}")
    print("History in_stock_dph: raw historical value, no lag shift")
    print("Future context excludes in_stock_dph")
    print("Future context includes distance_* calendar features")
    print("Integrated stock decoder: predicts future in_stock_dph_hat")
    print("Stock decoder safe mode: excludes future true total_dph and buy_box_dph")
    print("Safe historical DPH proxies: total_dph/buy_box_dph last/mean4/mean13")
    print(f"Context dim: {len(context_cols)}")
    return data, len(context_cols), context_cols


# =====================================================
# 2. Dataset
# =====================================================

class DemandDataset(Dataset):
    def __init__(self, data, history=52, horizon=20, mode="train", val_weeks=20):
        self.samples = []
        for asin, d in data.items():
            T = len(d["demand"])
            if mode == "train":
                starts = range(max(0, T - val_weeks - horizon - history + 1))
            else:
                s = T - history - horizon
                starts = [s] if s >= 0 else []

            for start in starts:
                self.samples.append({
                    "x": torch.tensor(d["features"][start:start+history], dtype=torch.float32),
                    "future_context": torch.tensor(
                        self._make_future_context_with_dph_proxies(
                            d=d,
                            start=start,
                            history=history,
                            horizon=horizon,
                        ),
                        dtype=torch.float32),
                    "y": torch.tensor(d["demand"][start+history:start+history+horizon], dtype=torch.float32),
                    "asin": asin,
                    "target_week": [str(w)[:10] for w in d["week"][start+history:start+history+horizon]],
                    "oos": torch.tensor(d["oos"][start+history:start+history+horizon], dtype=torch.float32),
                    "our_price": torch.tensor(
                        d["price_raw"][start+history:start+history+horizon],
                        dtype=torch.float32),
                    "pkg_volume": torch.tensor(
                        d["pkg_volume_raw"][start+history:start+history+horizon],
                        dtype=torch.float32),
                    "future_instock": torch.tensor(
                        d["instock_raw"][start+history:start+history+horizon],
                        dtype=torch.float32),
                })

    def _safe_hist_mean(self, arr, start, history, window):
        hist = arr[start:start+history]
        if len(hist) == 0:
            return 0.0
        hist = hist[-min(window, len(hist)):]
        return float(np.mean(hist))

    def _make_future_context_with_dph_proxies(self, d, start, history, horizon):
        """
        Fill historical DPH summary proxy features using only values up to forecast origin.
        These are repeated across the horizon and do not use future true DPH.
        """
        fc = d["future_context"][start+history:start+history+horizon].copy()
        idx = d.get("dph_proxy_context_idx", {})

        total_hist = d.get("total_dph_raw", None)
        buy_hist = d.get("buy_box_dph_raw", None)

        def fill(col, val):
            if col in idx:
                fc[:, idx[col]] = np.log1p(max(float(val), 0.0))

        if total_hist is not None:
            total_last = total_hist[start+history-1] if history > 0 else 0.0
            fill("hist_total_dph_last_log", total_last)
            fill("hist_total_dph_mean4_log", self._safe_hist_mean(total_hist, start, history, 4))
            fill("hist_total_dph_mean13_log", self._safe_hist_mean(total_hist, start, history, 13))

        if buy_hist is not None:
            buy_last = buy_hist[start+history-1] if history > 0 else 0.0
            fill("hist_buy_box_dph_last_log", buy_last)
            fill("hist_buy_box_dph_mean4_log", self._safe_hist_mean(buy_hist, start, history, 4))
            fill("hist_buy_box_dph_mean13_log", self._safe_hist_mean(buy_hist, start, history, 13))

        return fc

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


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
    def __init__(self, d_model=32, n_heads=4, beta_peak=1.0, soft_mask_scale=3.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.beta_peak = beta_peak
        self.soft_mask_scale = soft_mask_scale

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(0.1)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x, b_t, peak_score):
        B, T, D = x.shape
        q = self.q_proj(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        k = self.k_proj(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        v = self.v_proj(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2,-1)) / np.sqrt(self.d_head)

        # Softly down-weight zero-demand weeks.
        sparse_mask = (b_t == 0) & ~(b_t == 0).all(dim=1, keepdim=True)
        scores = scores - self.soft_mask_scale * sparse_mask.float()[:, None, None, :]

        peak_norm = peak_score / (peak_score.max(dim=1, keepdim=True)[0] + 1e-6)
        scores = scores + self.beta_peak * peak_norm[:, None, None, :]

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out  = torch.matmul(attn, v)
        out  = out.transpose(1,2).contiguous().view(B,T,D)
        out  = self.out_proj(out)
        return self.norm(x + out)


class TCNSparseAttnEncoder(nn.Module):
    def __init__(self, input_dim=25, d_model=32, horizon=20):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(input_dim, d_model)

        # Dilations include quarterly and annual scales.
        dilations = [1, 2, 4, 8, 13, 26, 52]
        self.convs = nn.ModuleList([CausalConv1d(d_model, d_model, 2, d) for d in dilations])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in dilations])

        self.sparse_attn = SparsePeakAttention(d_model, n_heads=4, beta_peak=1.0)
        self.final_norm  = nn.LayerNorm(d_model)

        self.base_head  = nn.Sequential(nn.Linear(d_model,64), nn.ReLU(), nn.Linear(64,horizon))
        self.alpha_head = nn.Sequential(nn.Linear(d_model,64), nn.ReLU(), nn.Linear(64,horizon))

    def forward(self, x):
        b_t        = x[:, :, 1]
        peak_score = torch.sqrt(torch.expm1(x[:,:,0]).clamp(min=0) + 1e-6)

        h = self.input_proj(x).permute(0,2,1)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h) + h
            h = h.permute(0,2,1)
            h = norm(h)
            h = F.gelu(h)
            h = h.permute(0,2,1)

        h   = self.sparse_attn(h.permute(0,2,1), b_t, peak_score)
        h_t = self.final_norm(h[:,-1,:])

        mu    = F.softplus(self.base_head(h_t))
        alpha = F.softplus(self.alpha_head(h_t)) + 1e-4
        return mu, alpha, h_t


class ContextZGenerator(nn.Module):
    def __init__(self, d_phi=32, context_dim=2, d_z=16, horizon=20):
        super().__init__()
        self.d_z = d_z
        self.net = nn.Sequential(
            nn.Linear(d_phi + horizon * context_dim, 64),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 2 * d_z)
        )

    def forward(self, phi, future_context):
        B   = phi.shape[0]
        ctx = future_context.reshape(B, -1)
        out = self.net(torch.cat([phi, ctx], dim=-1))
        z_mean, z_logstd = out.chunk(2, dim=-1)
        z_std = F.softplus(z_logstd) + 1e-4
        return z_mean, z_std


class Epinet(nn.Module):
    def __init__(self, d_phi=32, d_z=16, horizon=20, prior_scale=0.3):
        super().__init__()
        self.d_z = d_z; self.horizon = horizon; self.prior_scale = prior_scale
        self.learnable = nn.Sequential(
            nn.Linear(d_z+d_phi,64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2*horizon*d_z)
        )
        self.prior = nn.Sequential(
            nn.Linear(d_z+d_phi,64), nn.ReLU(),
            nn.Linear(64, 2*horizon*d_z)
        )
        for p in self.prior.parameters(): p.requires_grad = False

    def forward(self, phi, z):
        inp = torch.cat([z, phi], dim=-1)
        sl  = self.learnable(inp).view(-1, 2*self.horizon, self.d_z)
        sl  = torch.einsum("bhd,bd->bh", sl, z)
        sp  = self.prior(inp).view(-1, 2*self.horizon, self.d_z)
        sp  = torch.einsum("bhd,bd->bh", sp, z) * self.prior_scale
        out = sl + sp
        return out[:,:self.horizon], out[:,self.horizon:]



class StockDecoder(nn.Module):
    """
    Auxiliary decoder for future in_stock_dph_hat.

    It uses encoder state h_t and known future context
    (price + holiday + distance, no true future in_stock_dph)
    to predict future log1p(in_stock_dph).
    """
    def __init__(self, d_model, context_dim, horizon=20, hidden=64):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(d_model + context_dim + 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_t, future_context):
        B, H, C = future_context.shape
        h_rep = h_t.unsqueeze(1).expand(B, H, h_t.shape[-1])

        horizon_idx = torch.arange(H, device=future_context.device).float()
        horizon_idx = horizon_idx.view(1, H, 1).expand(B, H, 1) / max(H, 1)

        horizon_sin = torch.sin(2 * np.pi * horizon_idx)
        horizon_cos = torch.cos(2 * np.pi * horizon_idx)

        inp = torch.cat([h_rep, future_context, horizon_sin, horizon_cos], dim=-1)
        stock_log_hat = self.net(inp).squeeze(-1)
        stock_log_hat = F.softplus(stock_log_hat)  # log1p(in_stock_dph_hat), nonnegative
        return stock_log_hat


class TCN_ENN(nn.Module):
    def __init__(self, input_dim=25, context_dim=2, d_model=32,
                 d_z=16, horizon=20, prior_scale=0.3,
                 use_stock_decoder=True):
        super().__init__()
        self.d_z = d_z
        self.horizon = horizon
        self.use_stock_decoder = use_stock_decoder
        self.context_dim = context_dim

        self.encoder = TCNSparseAttnEncoder(input_dim, d_model, horizon)

        if use_stock_decoder:
            self.stock_decoder = StockDecoder(d_model, context_dim, horizon)
            z_context_dim = context_dim + 1  # add predicted log1p(in_stock_dph_hat)
        else:
            self.stock_decoder = None
            z_context_dim = context_dim

        self.z_generator = ContextZGenerator(d_model, z_context_dim, d_z, horizon)
        self.epinet = Epinet(d_model, d_z, horizon, prior_scale)

    def _augment_context_with_stock_hat(self, h_t, future_context):
        """
        Predict future stock_hat and append log1p stock_hat to future_context.

        Demand head sees predicted stock_hat only, never true future in_stock_dph.
        """
        if not self.use_stock_decoder:
            return future_context, None

        stock_log_hat = self.stock_decoder(h_t, future_context)  # [B, H]
        future_context_aug = torch.cat(
            [future_context, stock_log_hat.unsqueeze(-1)],
            dim=-1,
        )
        return future_context_aug, stock_log_hat

    def forward(self, x, future_context, nZ=8):
        mu_base, alpha_base, h_t = self.encoder(x)
        future_context_aug, stock_log_hat = self._augment_context_with_stock_hat(h_t, future_context)

        phi = h_t.detach()
        z_mean, z_std = self.z_generator(phi, future_context_aug)

        # z regularization
        z_reg = 0.001 * (z_mean**2 + z_std**2).mean()

        preds = []
        for _ in range(nZ):
            eps = torch.randn_like(z_mean)
            z = z_mean + z_std * eps
            mu_e, al_e = self.epinet(phi, z)
            mu = F.softplus(mu_base + mu_e)
            alpha = F.softplus(alpha_base + al_e) + 1e-4
            preds.append((mu, alpha))

        return preds, z_reg, stock_log_hat

    def predict(self, x, future_context, M=50, return_stock=False):
        self.eval()
        with torch.no_grad():
            mu_base, alpha_base, h_t = self.encoder(x)
            future_context_aug, stock_log_hat = self._augment_context_with_stock_hat(h_t, future_context)

            phi = h_t.detach()
            z_mean, z_std = self.z_generator(phi, future_context_aug)

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

        if return_stock:
            return p50, p70, stock_log_hat

        return p50, p70



# =====================================================
# 4. Loss
# =====================================================

def negbin_nll_elementwise(y, mu, alpha):
    eps = 1e-6
    r   = (1.0/alpha).clamp(min=eps)
    p   = (mu*alpha/(1+mu*alpha)).clamp(eps, 1-eps)
    return -(
        torch.lgamma(y+r) - torch.lgamma(r) - torch.lgamma(y+1)
        + r*torch.log(1-p) + y*torch.log(p)
    )


def tail_weighted_negbin_nll(y, mu, alpha, beta_tail=0.5):
    nll    = negbin_nll_elementwise(y, mu, alpha)
    weight = 1.0 + beta_tail * torch.log1p(y)
    return (nll * weight).sum() / weight.sum().clamp(min=1.0)


def pinball(y, pred, q):
    d = y - pred
    return torch.mean(torch.max(q*d, (q-1)*d))


def stock_decoder_loss(stock_log_hat, future_instock_true, mean_weight=0.10):
    """
    Huber loss on log1p(in_stock_dph), plus a small mean-calibration penalty.

    true future in_stock_dph is used only as an auxiliary supervision target.
    Demand prediction uses stock_log_hat, not true future in_stock_dph.

    The mean-calibration term helps prevent systematic underprediction or overprediction
    of the stock decoder scale.
    """
    if stock_log_hat is None:
        return torch.tensor(0.0, device=future_instock_true.device)

    target = torch.log1p(future_instock_true.clamp(min=0.0))

    point_loss = F.huber_loss(stock_log_hat, target, delta=1.0)

    pred_level = torch.expm1(stock_log_hat).clamp(min=0.0)
    true_level = future_instock_true.clamp(min=0.0)

    mean_loss = torch.abs(
        torch.log1p(pred_level.mean()) - torch.log1p(true_level.mean())
    )

    return point_loss + mean_weight * mean_loss




# =====================================================
# 5. Diagnostics
# =====================================================

def occurrence_probe_linear_nonlinear(h_ts, ys):
    """
    Probe whether future occurrence is linearly or nonlinearly readable from h_t.
    Targets:
      any_active: at least one positive demand in horizon
      next4_active: at least one positive demand in first 4 weeks
      active_rate_high: horizon active rate above median
    """
    targets = {
        "any_active": (ys > 0).any(axis=1),
        "next4_active": (ys[:, :min(4, ys.shape[1])] > 0).any(axis=1),
    }

    active_rate = (ys > 0).mean(axis=1)
    median_rate = np.median(active_rate)
    targets["active_rate_high"] = active_rate > median_rate

    rows = []

    for target_name, y_bin in targets.items():
        y_bin = y_bin.astype(int)

        if y_bin.sum() < 10 or (len(y_bin) - y_bin.sum()) < 10:
            rows.append({
                "target": target_name,
                "positive_rate": y_bin.mean(),
                "linear_auc": np.nan,
                "nonlinear_auc": np.nan,
                "nonlinear_gain": np.nan,
                "note": "skip: class imbalance",
            })
            continue

        try:
            linear_clf = LogisticRegression(max_iter=500, C=1.0)
            linear_clf.fit(h_ts, y_bin)
            linear_auc = roc_auc_score(y_bin, linear_clf.predict_proba(h_ts)[:, 1])
        except Exception:
            linear_auc = np.nan

        try:
            nonlinear_clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
            nonlinear_clf.fit(h_ts, y_bin)
            nonlinear_auc = roc_auc_score(y_bin, nonlinear_clf.predict_proba(h_ts)[:, 1])
        except Exception:
            nonlinear_auc = np.nan

        rows.append({
            "target": target_name,
            "positive_rate": y_bin.mean(),
            "linear_auc": linear_auc,
            "nonlinear_auc": nonlinear_auc,
            "nonlinear_gain": nonlinear_auc - linear_auc
                if np.isfinite(linear_auc) and np.isfinite(nonlinear_auc)
                else np.nan,
            "note": "",
        })

    out = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("OCCURRENCE PROBE: LINEAR VS NONLINEAR")
    print("=" * 60)
    print(out)

    print("\nHow to read:")
    print("  high linear AUC: occurrence signal is linearly readable from h_t")
    print("  nonlinear AUC >> linear AUC: h_t contains occurrence signal, but in nonlinear form")
    print("  both low: encoder may not capture occurrence well")

    return out



def diagnose_encoder(model, va_ld):
    """
    诊断 encoder（h_t）的质量：
    1. h_t 能区分活跃/非活跃样本的能力（AUC）
    2. h_t 对 magnitude 的预测力（R²）
    3. mu_base 和真实需求的对比
    """
    print("\n" + "="*60)
    print("ENCODER DIAGNOSIS")
    print("="*60)

    model.eval()
    h_ts, ys, mu_bases = [], [], []

    with torch.no_grad():
        for b in va_ld:
            mu_base, alpha_base, h_t = model.encoder(b["x"])
            h_ts.append(h_t.numpy())
            ys.append(b["y"].numpy())
            mu_bases.append(mu_base.numpy())

    h_ts     = np.concatenate(h_ts)      # [N, d_model]
    ys       = np.concatenate(ys)        # [N, horizon]
    mu_bases = np.concatenate(mu_bases)  # [N, horizon]

    occurrence_probe_df = occurrence_probe_linear_nonlinear(h_ts, ys)

    # 1. occurrence 判别能力
    has_active = (ys > 0).any(axis=1)
    if has_active.sum() > 10 and (~has_active).sum() > 10:
        try:
            clf = LogisticRegression(max_iter=500, C=1.0)
            clf.fit(h_ts, has_active.astype(int))
            auc = roc_auc_score(has_active, clf.predict_proba(h_ts)[:,1])
            print(f"h_t → occurrence AUC: {auc:.3f}")
            if auc < 0.6:
                print("  ← 差：encoder 对 occurrence 判别能力不足")
            elif auc < 0.75:
                print("  ← 一般：有改进空间")
            else:
                print("  ← 好：encoder 对 occurrence 有判别能力")
        except Exception as e:
            print(f"AUC 计算失败: {e}")

    # 2. magnitude 预测力
    active_mask  = (ys > 0).any(axis=1)
    y_mean_active = ys[active_mask].mean(axis=1)
    h_active      = h_ts[active_mask]

    if len(h_active) > 20:
        try:
            reg = Ridge()
            reg.fit(h_active, np.log1p(y_mean_active))
            r2  = r2_score(np.log1p(y_mean_active), reg.predict(h_active))
            print(f"h_t → log(magnitude) R²: {r2:.3f}")
            if r2 < 0.1:
                print("  ← 差：encoder 对 magnitude 几乎没有预测力")
            elif r2 < 0.3:
                print("  ← 一般：有改进空间")
            else:
                print("  ← 好：encoder 对 magnitude 有预测力")
        except Exception as e:
            print(f"R² 计算失败: {e}")

    # 3. mu_base vs 真实需求
    active_weeks_mask = ys > 0
    if active_weeks_mask.sum() > 0:
        true_mean  = ys[active_weeks_mask].mean()
        mu_mean    = mu_bases[active_weeks_mask].mean()
        print(f"\nActive weeks comparison:")
        print(f"  true demand mean : {true_mean:.2f}")
        print(f"  mu_base mean     : {mu_mean:.2f}")
        print(f"  ratio (mu/true)  : {mu_mean/max(true_mean,1e-8):.3f}")
        if mu_mean / max(true_mean, 1e-8) < 0.3:
            print("  ← mu_base 严重低估，magnitude 学习有问题")
        elif mu_mean / max(true_mean, 1e-8) < 0.7:
            print("  ← mu_base 偏低，有改进空间")
        else:
            print("  ← mu_base 合理")

    # 4. z 的质量
    z_means, z_stds = [], []
    with torch.no_grad():
        for b in va_ld:
            _, _, h_t = model.encoder(b["x"])
            phi = h_t.detach()

            # Stock-decoder version:
            # z_generator expects future_context augmented with predicted stock_hat.
            if hasattr(model, "_augment_context_with_stock_hat"):
                fc_for_z, _ = model._augment_context_with_stock_hat(h_t, b["future_context"])
            else:
                fc_for_z = b["future_context"]

            zm, zs = model.z_generator(phi, fc_for_z)
            z_means.append(zm.numpy())
            z_stds.append(zs.numpy())

    z_means = np.concatenate(z_means)
    z_stds  = np.concatenate(z_stds)
    print(f"\nz quality:")
    print(f"  z_mean abs mean : {np.abs(z_means).mean():.3f} (should be small)")
    print(f"  z_std mean      : {z_stds.mean():.3f} (should be ~1)")
    if z_stds.mean() > 3.0:
        print("  ← z_std 过大，后验扩张，joint prediction 不稳定")
    elif z_stds.mean() < 0.1:
        print("  ← z_std 过小，z 失去不确定性表达能力")
    else:
        print("  ← z_std 合理")

    print("="*60)


def diagnose_training_batch(b, preds, epoch, bi, n_diag_batches=3):
    """Print diagnostics for the first few batches."""
    if bi >= n_diag_batches:
        return
    y = b["y"]
    active_cnt = (y > 0).sum().item()
    total_cnt  = y.numel()
    mu_mean    = torch.stack([mu for mu, _ in preds], dim=0).mean().item()
    y_active_mean = y[y > 0].mean().item() if active_cnt > 0 else 0.0
    print(
        f"  [batch {bi}] active={active_cnt}/{total_cnt} "
        f"({100*active_cnt/total_cnt:.1f}%) "
        f"mu_mean={mu_mean:.2f} "
        f"y_active_mean={y_active_mean:.2f}"
    )


# =====================================================
# 6. Training
# =====================================================

def train(
    model,
    tr_ld,
    va_ld,
    epochs=60,
    nZ=8,
    lr=1e-3,
    lambda_q=0.05,
    beta_tail=0.5,
    patience=5,        # early stop
    lambda_z_reg=1.0,  # z regularization
    lambda_stock=0.05, # auxiliary stock decoder loss weight
):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    best_sd  = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0

        for bi, b in enumerate(tr_ld):
            x  = b["x"]
            fc = b["future_context"]
            y  = b["y"]

            preds, z_reg, stock_log_hat = model(x, fc, nZ=nZ)

            nll_loss = sum(
                tail_weighted_negbin_nll(y, mu, alpha, beta_tail=beta_tail)
                for mu, alpha in preds
            ) / nZ

            mu_stack   = torch.stack([mu for mu,_ in preds], dim=1)
            p50_train  = mu_stack.quantile(0.5, dim=1)
            p70_train  = mu_stack.quantile(0.7, dim=1)
            p70_train  = torch.maximum(p70_train, p50_train)
            q_loss     = pinball(y, p50_train, 0.5) + pinball(y, p70_train, 0.7)

            if "future_instock" in b:
                s_loss = stock_decoder_loss(stock_log_hat, b["future_instock"])
            else:
                s_loss = torch.tensor(0.0, device=y.device)

            loss = (
                nll_loss
                + lambda_q * q_loss
                + lambda_z_reg * z_reg
                + lambda_stock * s_loss
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()

            # Print batch diagnostics only in the first epoch.
            if epoch == 0:
                diagnose_training_batch(b, preds, epoch, bi)

        sch.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in va_ld:
                p50, p70 = model.predict(b["x"], b["future_context"], M=50)
                vl += (pinball(b["y"],p50,0.5) + pinball(b["y"],p70,0.7)).item()
        vl /= max(1, len(va_ld))

        improved = vl < best_val
        if improved:
            best_val = vl
            best_sd  = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch+1:3d} | "
            f"train={tr_loss/max(1,len(tr_ld)):.4f} | "
            f"val={vl:.4f} | "
            f"beta_tail={beta_tail} | stock_loss_w={lambda_stock}"
            + (" *" if improved else "")
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break

    if best_sd: model.load_state_dict(best_sd)
    print(f"Best val: {best_val:.4f}")


# =====================================================
# 7. Evaluation and forecast generation
# =====================================================

def evaluate(model, va_ld, M=100):
    all_y, all_p50, all_p70 = [], [], []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            p50, p70 = model.predict(b["x"], b["future_context"], M=M)
            all_y.append(b["y"].numpy())
            all_p50.append(p50.numpy())
            all_p70.append(p70.numpy())
    y   = np.concatenate(all_y)
    p50 = np.concatenate(all_p50)
    p70 = np.concatenate(all_p70)
    yt  = torch.tensor(y)
    return {
        "pinball50": pinball(yt, torch.tensor(p50), 0.5).item(),
        "pinball70": pinball(yt, torch.tensor(p70), 0.7).item(),
    }


def generate_forecast_df(model, va_ld, M=50):
    rows = []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            p50, p70, stock_log_hat = model.predict(b["x"], b["future_context"], M=M, return_stock=True)
            hist_mean = (b["x"][:,:,0].exp()-1).mean(dim=1,keepdim=True).clamp(min=0)
            hm50 = hist_mean.expand_as(b["y"])
            hm70 = hm50 * 1.25
            for i in range(b["y"].shape[0]):
                for h in range(b["y"].shape[1]):
                    rows.append({
                        "asin": b["asin"][i],
                        "order_week": pd.to_datetime(b["target_week"][h][i]),
                        "fcst_week_index": h+1,
                        "fbi_demand": b["y"][i,h].item(),
                        "our_price": b["our_price"][i,h].item(),
                        "true_amt": b["y"][i,h].item() * b["our_price"][i,h].item(),
                        "pkg_volume": b["pkg_volume"][i,h].item(),
                        "true_size": b["y"][i,h].item() * b["pkg_volume"][i,h].item(),
                        "true_future_instock": b["future_instock"][i,h].item()
                            if "future_instock" in b else np.nan,
                        "pred_instock_dph_hat": torch.expm1(stock_log_hat[i,h]).item()
                            if stock_log_hat is not None else np.nan,
                        "pred_instock_log_hat": stock_log_hat[i,h].item()
                            if stock_log_hat is not None else np.nan,
                        "scot_oos": b["oos"][i,h].item(),
                        "oos": b["oos"][i,h].item(),
                        "oos_status": b["oos"][i,h].item(),
                        "p50_amxl": p50[i,h].item(),
                        "p70_amxl": p70[i,h].item(),
                        "p50_scot": hm50[i,h].item(),
                        "p70_scot": hm70[i,h].item(),
                    })
    return pd.DataFrame(rows)


def generate_diagnostic_df(model, va_ld, M=100, threshold=0.5):
    rows = []
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            p50, p70 = model.predict(b["x"], b["future_context"], M=M)
            for i in range(b["y"].shape[0]):
                for h in range(b["y"].shape[1]):
                    y_val   = b["y"][i,h].item()
                    p50_val = p50[i,h].item()
                    p70_val = p70[i,h].item()
                    rows.append({
                        "asin": b["asin"][i],
                        "order_week": pd.to_datetime(b["target_week"][h][i]),
                        "horizon": h+1,
                        "y": y_val, "p50": p50_val, "p70": p70_val,
                        "true_active": int(y_val > 0),
                        "pred_active_p50": int(p50_val > threshold),
                        "pred_active_p70": int(p70_val > threshold),
                    })
    return pd.DataFrame(rows)


def underbias_diagnosis(diag_df, pred_col="p70", threshold=0.5):
    y    = diag_df["y"].values
    pred = diag_df[pred_col].values
    ta   = y > 0
    pa   = pred > threshold
    tp = np.sum(ta & pa); fp = np.sum(~ta & pa)
    fn = np.sum(ta & ~pa); tn = np.sum(~ta & ~pa)
    recall    = tp / max(1, tp+fn)
    precision = tp / max(1, tp+fp)
    f1        = 2*precision*recall / max(1e-8, precision+recall)
    total_under = np.maximum(y-pred, 0).sum()
    missed_under    = np.maximum(y[ta & ~pa] - pred[ta & ~pa], 0).sum()
    magnitude_under = np.maximum(y[ta & pa]  - pred[ta & pa],  0).sum()
    ratio = pred[ta & pa] / np.maximum(y[ta & pa], 1e-8) if (ta & pa).sum() > 0 else np.array([np.nan])
    return pd.DataFrame([{
        "pred_col": pred_col, "threshold": threshold,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "occurrence_recall": recall, "occurrence_precision": precision, "occurrence_f1": f1,
        "total_underbias": total_under,
        "underbias_rate": total_under / max(1e-8, y.sum()),
        "missed_active_share": missed_under / max(1e-8, total_under),
        "magnitude_under_share": magnitude_under / max(1e-8, total_under),
        "avg_pred_over_true_when_active_predicted": np.nanmean(ratio),
        "median_pred_over_true_when_active_predicted": np.nanmedian(ratio),
    }])


def magnitude_gap(diag_df):
    df = diag_df[diag_df["true_active"]==1].copy()
    if len(df) == 0: return pd.DataFrame()
    y, p50, p70 = df["y"].values, df["p50"].values, df["p70"].values
    out = pd.DataFrame([{
        "true_active_mean": y.mean(),
        "p50_active_mean": p50.mean(),
        "p70_active_mean": p70.mean(),
        "p50_pct_of_true": p50.mean()/max(y.mean(),1e-8),
        "p70_pct_of_true": p70.mean()/max(y.mean(),1e-8),
        "p50_gap": y.mean()-p50.mean(),
        "p70_gap": y.mean()-p70.mean(),
    }])
    print("\n[Magnitude Gap - Active weeks only]")
    print(out.T)
    return out


# =====================================================
# 8. Run
# =====================================================

def filter_extreme_asins(data_high, demand_col="fbi_demand", asin_col="asin", q=0.99):
    df = data_high.copy()
    df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce").fillna(0).clip(lower=0)
    pos = df.loc[df[demand_col]>0, demand_col]
    if len(pos) == 0: return df, pd.DataFrame(), np.nan
    cap = float(pos.quantile(q))
    asin_peak = df.groupby(asin_col)[demand_col].max().reset_index(name="asin_max")
    bad_asins = asin_peak.loc[asin_peak["asin_max"]>cap, asin_col]
    clean = df[~df[asin_col].isin(bad_asins)].copy()
    print(f"\nExtreme ASIN filter (p{int(q*100)}={cap:.1f}): removed {bad_asins.nunique()} ASINs")
    print(f"Clean ASINs: {clean[asin_col].nunique()} | Clean rows: {len(clean)}")
    return clean, asin_peak[asin_peak[asin_col].isin(bad_asins)], cap


def run_nb_high_sparse(
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
    M_eval=100,
    lambda_q=0.05,
    beta_tail=0.5,
    patience=5,
    lambda_z_reg=1.0,
    lambda_stock=0.05,
    remove_extreme=True,
    extreme_q=0.99,
):
    print("="*70)
    print("NB-v2 HIGH-SPARSE | leak-fix + soft-mask + dilation13 + early-stop + z-reg")
    print("="*70)

    data_small, _ = add_zero_rate_group(
        prepare_data_sample(data_raw1, n_asins), zero_thresholds
    )
    data_high = data_small[data_small["zero_group"]=="high_sparse"].copy()

    if remove_extreme:
        data_high, _, _ = filter_extreme_asins(data_high, q=extreme_q)

    data, context_dim, context_cols = load_real_data(data_high)
    all_demand = np.concatenate([d["demand"] for d in data.values()])
    print(f"ASINs: {len(data)} | Zero rate: {(all_demand==0).mean():.1%}")

    tr_ds = DemandDataset(data, history, horizon, "train", horizon)
    va_ds = DemandDataset(data, history, horizon, "val",   horizon)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    print(f"Train: {len(tr_ds)} | Val: {len(va_ds)}")

    model = TCN_ENN(25, context_dim, d_model, d_z, horizon, prior_scale)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,} | d_model={d_model} | d_z={d_z}")
    print(f"beta_tail={beta_tail} | lambda_q={lambda_q} | patience={patience}")

    train(model, tr_ld, va_ld,
          epochs=epochs, nZ=8, lr=1e-3,
          lambda_q=lambda_q, beta_tail=beta_tail,
          patience=patience, lambda_z_reg=lambda_z_reg, lambda_stock=lambda_stock)

    # Encoder diagnostics.
    diagnose_encoder(model, va_ld)

    metrics = evaluate(model, va_ld, M=M_eval)
    print(f"\nPinball50={metrics['pinball50']:.4f} | Pinball70={metrics['pinball70']:.4f}")

    forecast_df = generate_forecast_df(model, va_ld, M=M_eval)
    forecast_df["zero_group_run"] = "high_sparse_nb_v2"

    diag_df  = generate_diagnostic_df(model, va_ld, M=M_eval)
    diag_p50 = underbias_diagnosis(diag_df, "p50")
    diag_p70 = underbias_diagnosis(diag_df, "p70")
    mag_gap_df = magnitude_gap(diag_df)

    print("\nUnderbias P50:"); print(diag_p50.T)
    print("\nUnderbias P70:"); print(diag_p70.T)

    return {
        "model": model,
        "forecast_df": forecast_df,
        "diag_df": diag_df,
        "diag_p50": diag_p50,
        "diag_p70": diag_p70,
        "mag_gap": mag_gap_df,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
    }



# =====================================================
# 9. Final WAPE summary
# =====================================================

def run_final_wape(result, remove_oos_dp=True, source="lp"):
    """
    Compute final boss-style WAPE from result["forecast_df"].

    This function expects these notebook functions to already exist:
      - calculate_wape_using_lp_oos2
      - quick_error_check
    """
    if "forecast_df" not in result:
        raise KeyError('result must contain "forecast_df".')

    if "calculate_wape_using_lp_oos2" not in globals():
        raise RuntimeError("calculate_wape_using_lp_oos2 is not defined.")

    if "quick_error_check" not in globals():
        raise RuntimeError("quick_error_check is not defined.")

    forecast_df = result["forecast_df"]

    wape_df = calculate_wape_using_lp_oos2(
        forecast_df,
        [0.5, 0.7],
        remove_oos_dp=remove_oos_dp,
        source=source,
    )

    cols_p50 = [
        "p50_amxl_penalty",
        "p50_scot_penalty",
        "p50_amxl_overbias",
        "p50_scot_overbias",
        "p50_amxl_underbias",
        "p50_scot_underbias",
        "fbi_demand",
    ]

    cols_p70 = [
        "p70_amxl_penalty",
        "p70_scot_penalty",
        "p70_amxl_overbias",
        "p70_scot_overbias",
        "p70_amxl_underbias",
        "p70_scot_underbias",
        "fbi_demand",
    ]

    p50_wape, p50_penalty_diff = quick_error_check(wape_df, cols_p50)
    p70_wape, p70_penalty_diff = quick_error_check(wape_df, cols_p70)

    print("\n" + "=" * 80)
    print("FINAL WAPE SUMMARY")
    print("=" * 80)

    print("\nP50 WAPE")
    print(p50_wape)
    print("P50 penalty diff:", p50_penalty_diff)

    print("\nP70 WAPE")
    print(p70_wape)
    print("P70 penalty diff:", p70_penalty_diff)

    return {
        "wape_df": wape_df,
        "p50_wape": p50_wape,
        "p70_wape": p70_wape,
        "p50_penalty_diff": p50_penalty_diff,
        "p70_penalty_diff": p70_penalty_diff,
    }


def run_nb_high_sparse_with_wape(
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
    M_eval=100,
    lambda_q=0.05,
    beta_tail=0.5,
    patience=5,
    lambda_z_reg=1.0,
    lambda_stock=0.05,
    remove_extreme=True,
    extreme_q=0.99,
    remove_oos_dp=True,
):
    """
    Run the full experiment and print final WAPE.
    """
    result = run_nb_high_sparse(
        data_raw1=data_raw1,
        n_asins=n_asins,
        zero_thresholds=zero_thresholds,
        prior_scale=prior_scale,
        epochs=epochs,
        history=history,
        horizon=horizon,
        d_model=d_model,
        d_z=d_z,
        batch_size=batch_size,
        M_eval=M_eval,
        lambda_q=lambda_q,
        beta_tail=beta_tail,
        patience=patience,
        lambda_z_reg=lambda_z_reg,
        remove_extreme=remove_extreme,
        extreme_q=extreme_q,
    )

    wape_outputs = run_final_wape(
        result,
        remove_oos_dp=remove_oos_dp,
        source="lp",
    )

    result["wape_outputs"] = wape_outputs

    return result



# =====================================================
# 10. Sparse-group WAPE diagnostics
# =====================================================

def attach_zero_group_to_joined_df(joined_df, asin_stats):
    """
    Attach zero_rate and zero_group to the joined AMXL-SCOT forecast dataframe.
    """
    if asin_stats is None or len(asin_stats) == 0:
        return joined_df.copy()

    out = joined_df.copy()
    stats = asin_stats.copy()

    out["asin"] = out["asin"].astype(str)
    stats["asin"] = stats["asin"].astype(str)

    keep = [c for c in ["asin", "zero_rate", "zero_group"] if c in stats.columns]

    if "zero_group" not in keep:
        return out

    out = out.merge(
        stats[keep].drop_duplicates("asin"),
        on="asin",
        how="left",
    )

    return out


def summarize_wape_by_sparse_group(wape_df, joined_df_with_group):
    """
    Summarize boss-style WAPE by zero_group using the already-generated wape_df.
    This is diagnostic only; the main result remains the overall WAPE.
    """
    if "zero_group" not in joined_df_with_group.columns:
        print("zero_group not found. Skip sparse-group WAPE diagnostics.")
        return pd.DataFrame()

    key_cols = ["asin", "order_week", "zero_rate", "zero_group"]
    group_map = joined_df_with_group[key_cols].drop_duplicates(["asin", "order_week"]).copy()

    work = wape_df.copy()
    work["asin"] = work["asin"].astype(str)
    work["order_week"] = pd.to_datetime(work["order_week"])
    group_map["asin"] = group_map["asin"].astype(str)
    group_map["order_week"] = pd.to_datetime(group_map["order_week"])

    work = work.merge(group_map, on=["asin", "order_week"], how="left")

    total_demand_all = work["fbi_demand"].sum()
    total_rows_all = len(work)
    total_asins_all = work["asin"].nunique()

    rows = []

    for group_name, g in work.groupby("zero_group", dropna=False):
        denom = g["fbi_demand"].sum()

        rows.append({
            "zero_group": group_name,
            "n_rows": len(g),
            "n_asins": g["asin"].nunique(),
            "total_fbi_demand": denom,
            "true_mean": g["fbi_demand"].mean(),
            "p50_amxl_penalty": g["p50_amxl_penalty"].sum() / denom if denom > 0 else np.nan,
            "p50_scot_penalty": g["p50_scot_penalty"].sum() / denom if denom > 0 else np.nan,
            "p50_bps_improvement": (
                (g["p50_scot_penalty"].sum() - g["p50_amxl_penalty"].sum()) / denom * 10000
                if denom > 0 else np.nan
            ),
            "p70_amxl_penalty": g["p70_amxl_penalty"].sum() / denom if denom > 0 else np.nan,
            "p70_scot_penalty": g["p70_scot_penalty"].sum() / denom if denom > 0 else np.nan,
            "p70_bps_improvement": (
                (g["p70_scot_penalty"].sum() - g["p70_amxl_penalty"].sum()) / denom * 10000
                if denom > 0 else np.nan
            ),
            "p50_amxl_underbias": g["p50_amxl_underbias"].sum() / denom if denom > 0 else np.nan,
            "p50_scot_underbias": g["p50_scot_underbias"].sum() / denom if denom > 0 else np.nan,
            "p50_amxl_overbias": g["p50_amxl_overbias"].sum() / denom if denom > 0 else np.nan,
            "p50_scot_overbias": g["p50_scot_overbias"].sum() / denom if denom > 0 else np.nan,
            "p70_amxl_underbias": g["p70_amxl_underbias"].sum() / denom if denom > 0 else np.nan,
            "p70_scot_underbias": g["p70_scot_underbias"].sum() / denom if denom > 0 else np.nan,
            "p70_amxl_overbias": g["p70_amxl_overbias"].sum() / denom if denom > 0 else np.nan,
            "p70_scot_overbias": g["p70_scot_overbias"].sum() / denom if denom > 0 else np.nan,
        })

    out = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("SPARSE-GROUP WAPE DIAGNOSTICS")
    print("=" * 80)

    display_cols = [
        "zero_group",
        "n_asins",
        "n_rows",
        "total_fbi_demand",
        "total_amt",
        "total_size",
        "demand_share",
        "avg_total_demand_per_asin",
        "true_mean",
        "true_zero_rate",
        "p50_amxl_penalty",
        "p50_scot_penalty",
        "p50_bps_improvement",
        "p70_amxl_penalty",
        "p70_scot_penalty",
        "p70_bps_improvement",
        "p50_amxl_underbias",
        "p50_scot_underbias",
        "p50_amxl_overbias",
        "p50_scot_overbias",
        "p70_amxl_underbias",
        "p70_scot_underbias",
        "p70_amxl_overbias",
        "p70_scot_overbias",
    ]
    display_cols = [c for c in display_cols if c in out.columns]
    print(out[display_cols])

    return out


# =====================================================
# 10. Real SCOT alignment and WAPE
# =====================================================

def run_high_sparse_scot_alignment_wape(
    result,
    scot_df,
    data_raw1=None,
    asin_stats=None,
    remove_oos_dp=True,
    source="lp",
):
    """
    Align real SCOT forecasts to result["forecast_df"] and compute WAPE.
    """
    if "calculate_wape_using_lp_oos2" not in globals():
        raise RuntimeError("calculate_wape_using_lp_oos2 is not defined.")

    if "quick_error_check" not in globals():
        raise RuntimeError("quick_error_check is not defined.")

    forecast_df = result["forecast_df"].copy()
    forecast_df.columns = [c.strip() for c in forecast_df.columns]
    forecast_df["asin"] = forecast_df["asin"].astype(str)
    forecast_df["order_week"] = pd.to_datetime(forecast_df["order_week"])

    scot = scot_df.copy()
    scot.columns = [c.strip() for c in scot.columns]

    for c in ["asin", "order_week", "forecast_qty_p50", "forecast_qty_p70"]:
        if c not in scot.columns:
            raise ValueError(f"Missing SCOT column: {c}")

    scot["asin"] = scot["asin"].astype(str)
    scot["order_week"] = pd.to_datetime(scot["order_week"])
    scot["forecast_qty_p50"] = pd.to_numeric(scot["forecast_qty_p50"], errors="coerce")
    scot["forecast_qty_p70"] = pd.to_numeric(scot["forecast_qty_p70"], errors="coerce")

    if "fcst_start_week" in scot.columns:
        scot["fcst_start_week"] = pd.to_datetime(scot["fcst_start_week"])

    print("\n" + "=" * 80)
    print("NB FORECAST WINDOW")
    print("=" * 80)
    print("NB rows:", len(forecast_df))
    print("NB ASINs:", forecast_df["asin"].nunique())
    print("NB weeks:", forecast_df["order_week"].min(), "to", forecast_df["order_week"].max())
    print("NB week count:", forecast_df["order_week"].nunique())

    print("\n" + "=" * 80)
    print("REAL SCOT FORECAST FILE")
    print("=" * 80)
    print("SCOT rows:", len(scot))
    print("SCOT ASINs:", scot["asin"].nunique())
    print("SCOT weeks:", scot["order_week"].min(), "to", scot["order_week"].max())
    print("SCOT week count:", scot["order_week"].nunique())

    if "fcst_start_week" in scot.columns:
        print("\nSCOT fcst_start_week counts:")
        print(scot["fcst_start_week"].value_counts().sort_index())

    scot_keep = (
        scot[["asin", "order_week", "forecast_qty_p50", "forecast_qty_p70"]]
        .groupby(["asin", "order_week"], as_index=False)
        .agg(
            forecast_qty_p50=("forecast_qty_p50", "mean"),
            forecast_qty_p70=("forecast_qty_p70", "mean"),
        )
    )

    forecast_df_scot_real = forecast_df.merge(
        scot_keep,
        on=["asin", "order_week"],
        how="inner",
    )

    row_match_rate = len(forecast_df_scot_real) / max(len(forecast_df), 1)
    asin_match_rate = (
        forecast_df_scot_real["asin"].nunique()
        / max(forecast_df["asin"].nunique(), 1)
    )

    print("\n" + "=" * 80)
    print("ALIGNMENT CHECK")
    print("=" * 80)
    print("NB forecast rows:", len(forecast_df))
    print("After SCOT merge rows:", len(forecast_df_scot_real))
    print("Matched ASINs:", forecast_df_scot_real["asin"].nunique())
    print("Matched weeks:", forecast_df_scot_real["order_week"].min(), "to",
          forecast_df_scot_real["order_week"].max())
    print("Matched week count:", forecast_df_scot_real["order_week"].nunique())
    print("Row match rate:", row_match_rate)
    print("ASIN match rate:", asin_match_rate)

    print("\n" + "=" * 80)
    print("ASIN SELECTION CHECK")
    print("=" * 80)
    print("Selected NB ASINs:", forecast_df["asin"].nunique())
    print("Matched ASINs with SCOT:", forecast_df_scot_real["asin"].nunique())
    print(
        "Missing ASINs after SCOT merge:",
        forecast_df["asin"].nunique() - forecast_df_scot_real["asin"].nunique(),
    )

    forecast_df_scot_real["p50_scot"] = forecast_df_scot_real["forecast_qty_p50"]
    forecast_df_scot_real["p70_scot"] = np.maximum(
        forecast_df_scot_real["forecast_qty_p70"],
        forecast_df_scot_real["forecast_qty_p50"],
    )

    mean_check = pd.DataFrame([{
        "n_rows": len(forecast_df_scot_real),
        "n_asins": forecast_df_scot_real["asin"].nunique(),
        "true_mean": forecast_df_scot_real["fbi_demand"].mean(),
        "total_amt": (
            forecast_df_scot_real["true_amt"].sum()
            if "true_amt" in forecast_df_scot_real.columns
            else np.nan
        ),
        "total_size": (
            forecast_df_scot_real["true_size"].sum()
            if "true_size" in forecast_df_scot_real.columns
            else np.nan
        ),
        "amxl_p50_mean": forecast_df_scot_real["p50_amxl"].mean(),
        "amxl_p70_mean": forecast_df_scot_real["p70_amxl"].mean(),
        "real_scot_p50_mean": forecast_df_scot_real["p50_scot"].mean(),
        "real_scot_p70_mean": forecast_df_scot_real["p70_scot"].mean(),
        "true_zero_rate": (forecast_df_scot_real["fbi_demand"] == 0).mean(),
        "true_active_ratio": (forecast_df_scot_real["fbi_demand"] > 0).mean(),
    }])

    print("\n" + "=" * 80)
    print("FORECAST MEAN CHECK")
    print("=" * 80)
    print(mean_check.T)

    wape_df = calculate_wape_using_lp_oos2(
        forecast_df_scot_real,
        [0.5, 0.7],
        remove_oos_dp=remove_oos_dp,
        source=source,
    )

    if asin_stats is None and "asin_stats" in result:
        asin_stats = result["asin_stats"]

    forecast_df_scot_real_with_group = attach_zero_group_to_joined_df(
        forecast_df_scot_real,
        asin_stats,
    )

    sparse_group_wape = summarize_wape_by_sparse_group(
        wape_df,
        forecast_df_scot_real_with_group,
    )

    cols_p50 = [
        "p50_amxl_penalty", "p50_scot_penalty",
        "p50_amxl_overbias", "p50_scot_overbias",
        "p50_amxl_underbias", "p50_scot_underbias",
        "fbi_demand",
    ]

    cols_p70 = [
        "p70_amxl_penalty", "p70_scot_penalty",
        "p70_amxl_overbias", "p70_scot_overbias",
        "p70_amxl_underbias", "p70_scot_underbias",
        "fbi_demand",
    ]

    p50_wape, p50_penalty_diff = quick_error_check(wape_df, cols_p50)
    p70_wape, p70_penalty_diff = quick_error_check(wape_df, cols_p70)

    print("\n" + "=" * 80)
    print("FINAL WAPE WITH REAL SCOT")
    print("=" * 80)
    print("\nP50 WAPE:")
    print(p50_wape)
    print("P50 penalty diff AMXL - SCOT:", p50_penalty_diff)
    print("\nP70 WAPE:")
    print(p70_wape)
    print("P70 penalty diff AMXL - SCOT:", p70_penalty_diff)

    return {
        "forecast_df_scot_real": forecast_df_scot_real,
        "forecast_df_scot_real_with_group": forecast_df_scot_real_with_group,
        "wape_df": wape_df,
        "sparse_group_wape": sparse_group_wape,
        "mean_check": mean_check,
        "p50_wape": p50_wape,
        "p70_wape": p70_wape,
        "p50_penalty_diff": p50_penalty_diff,
        "p70_penalty_diff": p70_penalty_diff,
    }


# =====================================================
# 11. Train on sample-SCOT intersection
# =====================================================

def run_nb_high_sparse_from_sample_scot_intersection(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
    zero_thresholds=(0.4, 0.7),
    prior_scale=0.3,
    epochs=60,
    history=52,
    horizon=20,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    lambda_q=0.05,
    beta_tail=0.5,
    patience=5,
    lambda_z_reg=1.0,
    lambda_stock=0.05,
    remove_extreme=True,
    extreme_q=0.99,
    run_wape=True,
    remove_oos_dp=True,
):
    """
    Sample 5000 from data_raw1, keep SCOT intersection, train high_sparse, and compute WAPE.
    """
    print("=" * 80)
    print("LEGACY NB HIGH-SPARSE | SAMPLE 5000 THEN KEEP SCOT INTERSECTION")
    print("=" * 80)

    data_small_raw, sample_asin_df, intersect_asin_df = (
        prepare_data_from_sample_scot_intersection(
            data_raw1=data_raw1,
            scot_df=scot_df,
            n_asins=n_asins,
            seed=seed,
        )
    )

    data_small, asin_stats = add_zero_rate_group(data_small_raw, zero_thresholds)
    data_high = data_small[data_small["zero_group"] == "high_sparse"].copy()

    print("\n" + "=" * 80)
    print("HIGH-SPARSE AFTER SCOT INTERSECTION")
    print("=" * 80)
    print("High-sparse ASINs:", data_high["asin"].nunique())
    print("High-sparse rows:", len(data_high))

    if remove_extreme:
        data_high, removed_extreme, extreme_cap = filter_extreme_asins(
            data_high,
            q=extreme_q,
        )
    else:
        removed_extreme = pd.DataFrame()
        extreme_cap = np.nan

    data, context_dim, context_cols = load_real_data(data_high)

    all_demand = np.concatenate([d["demand"] for d in data.values()])
    print(f"ASINs used for training: {len(data)}")
    print(f"Zero rate: {(all_demand == 0).mean():.1%}")

    tr_ds = DemandDataset(data, history, horizon, "train", horizon)
    va_ds = DemandDataset(data, history, horizon, "val", horizon)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(tr_ds)} | Val samples: {len(va_ds)}")

    model = TCN_ENN(
        input_dim=25,
        context_dim=context_dim,
        d_model=d_model,
        d_z=d_z,
        horizon=horizon,
        prior_scale=prior_scale,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,} | d_model={d_model} | d_z={d_z}")
    print(f"beta_tail={beta_tail} | lambda_q={lambda_q} | patience={patience}")

    train(
        model,
        tr_ld,
        va_ld,
        epochs=epochs,
        nZ=8,
        lr=1e-3,
        lambda_q=lambda_q,
        beta_tail=beta_tail,
        patience=patience,
        lambda_z_reg=lambda_z_reg,
        lambda_stock=lambda_stock,
    )

    diagnose_encoder(model, va_ld)

    metrics = evaluate(model, va_ld, M=M_eval)
    print(f"\nPinball50={metrics['pinball50']:.4f} | Pinball70={metrics['pinball70']:.4f}")

    forecast_df = generate_forecast_df(model, va_ld, M=M_eval)
    forecast_df["zero_group_run"] = "high_sparse_sample_scot_intersection"

    diag_df = generate_diagnostic_df(model, va_ld, M=M_eval)
    diag_p50 = underbias_diagnosis(diag_df, "p50")
    diag_p70 = underbias_diagnosis(diag_df, "p70")
    mag_gap_df = magnitude_gap(diag_df)

    print("\nUnderbias P50:")
    print(diag_p50.T)
    print("\nUnderbias P70:")
    print(diag_p70.T)

    result = {
        "model": model,
        "forecast_df": forecast_df,
        "diag_df": diag_df,
        "diag_p50": diag_p50,
        "diag_p70": diag_p70,
        "mag_gap": mag_gap_df,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
        "data_small": data_small,
        "data_high": data_high,
        "asin_stats": asin_stats,
        "sample_asin_df": sample_asin_df,
        "intersect_asin_df": intersect_asin_df,
        "removed_extreme": removed_extreme,
        "extreme_cap": extreme_cap,
    }

    result["stock_decoder_diag"] = diagnose_stock_decoder(result)

    if run_wape:
        result["real_scot_outputs"] = run_high_sparse_scot_alignment_wape(
            result=result,
            scot_df=scot_df,
            data_raw1=data_raw1,
            asin_stats=asin_stats,
            remove_oos_dp=remove_oos_dp,
            source="lp",
        )

    return result



# =====================================================
# 12. Train on all sample-SCOT intersection ASINs
# =====================================================

def run_nb_all_sample_scot_intersection(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
    zero_thresholds=(0.4, 0.7),
    prior_scale=0.3,
    epochs=60,
    history=52,
    horizon=20,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    lambda_q=0.05,
    beta_tail=0.5,
    patience=5,
    lambda_z_reg=1.0,
    lambda_stock=0.05,
    remove_extreme=True,
    extreme_q=0.99,
    run_wape=True,
    remove_oos_dp=True,
):
    """
    Main experiment:
      1. sample 5000 ASINs from data_raw1
      2. keep ASINs also present in scot_df
      3. assign sparse labels for diagnostics only
      4. train one model on all intersection ASINs
      5. align with real SCOT and compute overall + sparse-group WAPE
    """
    print("=" * 80)
    print("NB ALL-ASIN | SAMPLE 5000 THEN KEEP SCOT INTERSECTION")
    print("=" * 80)

    data_intersection_raw, sample_asin_df, intersect_asin_df = (
        prepare_data_from_sample_scot_intersection(
            data_raw1=data_raw1,
            scot_df=scot_df,
            n_asins=n_asins,
            seed=seed,
        )
    )

    # Sparse labels are for diagnostics only. No filtering by group.
    data_labeled, asin_stats = add_zero_rate_group(
        data_intersection_raw,
        zero_thresholds,
    )

    print("\n" + "=" * 80)
    print("TRAINING SET AFTER SCOT INTERSECTION")
    print("=" * 80)
    print("Training ASINs:", data_labeled["asin"].nunique())
    print("Training rows:", len(data_labeled))

    print("\nSparse-group labels for diagnostics only:")
    print(
        data_labeled
        .groupby("zero_group")["asin"]
        .nunique()
        .reset_index(name="n_asins")
    )

    data_train = data_labeled.copy()

    if remove_extreme:
        data_train, removed_extreme, extreme_cap = filter_extreme_asins(
            data_train,
            q=extreme_q,
        )
    else:
        removed_extreme = pd.DataFrame()
        extreme_cap = np.nan

    data, context_dim, context_cols = load_real_data(data_train)

    all_demand = np.concatenate([d["demand"] for d in data.values()])
    print(f"ASINs used for training: {len(data)}")
    print(f"Overall zero rate: {(all_demand == 0).mean():.1%}")

    tr_ds = DemandDataset(data, history, horizon, "train", horizon)
    va_ds = DemandDataset(data, history, horizon, "val", horizon)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(tr_ds)} | Val samples: {len(va_ds)}")

    model = TCN_ENN(
        input_dim=25,
        context_dim=context_dim,
        d_model=d_model,
        d_z=d_z,
        horizon=horizon,
        prior_scale=prior_scale,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,} | d_model={d_model} | d_z={d_z}")
    print(f"beta_tail={beta_tail} | lambda_q={lambda_q} | patience={patience}")

    train(
        model,
        tr_ld,
        va_ld,
        epochs=epochs,
        nZ=8,
        lr=1e-3,
        lambda_q=lambda_q,
        beta_tail=beta_tail,
        patience=patience,
        lambda_z_reg=lambda_z_reg,
        lambda_stock=lambda_stock,
    )

    diagnose_encoder(model, va_ld)

    metrics = evaluate(model, va_ld, M=M_eval)
    print(f"\nPinball50={metrics['pinball50']:.4f} | Pinball70={metrics['pinball70']:.4f}")

    forecast_df = generate_forecast_df(model, va_ld, M=M_eval)
    forecast_df["zero_group_run"] = "all_sample_scot_intersection"

    diag_df = generate_diagnostic_df(model, va_ld, M=M_eval)
    diag_p50 = underbias_diagnosis(diag_df, "p50")
    diag_p70 = underbias_diagnosis(diag_df, "p70")
    mag_gap_df = magnitude_gap(diag_df)

    print("\nUnderbias P50:")
    print(diag_p50.T)

    print("\nUnderbias P70:")
    print(diag_p70.T)

    result = {
        "model": model,
        "forecast_df": forecast_df,
        "diag_df": diag_df,
        "diag_p50": diag_p50,
        "diag_p70": diag_p70,
        "mag_gap": mag_gap_df,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
        "data_intersection_raw": data_intersection_raw,
        "data_labeled": data_labeled,
        "data_train": data_train,
        "asin_stats": asin_stats,
        "sample_asin_df": sample_asin_df,
        "intersect_asin_df": intersect_asin_df,
        "removed_extreme": removed_extreme,
        "extreme_cap": extreme_cap,
    }

    if run_wape:
        result["real_scot_outputs"] = run_high_sparse_scot_alignment_wape(
            result=result,
            scot_df=scot_df,
            data_raw1=data_raw1,
            asin_stats=asin_stats,
            remove_oos_dp=remove_oos_dp,
            source="lp",
        )

    return result



# =====================================================
# 13. Final diagnostic printer
# =====================================================

def print_final_diagnostics(result):
    """
    Print the final joined dataframe shape and sparse-group diagnostic table.
    """
    outputs = result.get("real_scot_outputs", {})
    joined = outputs.get("forecast_df_scot_real_with_group", pd.DataFrame())
    sparse_diag = outputs.get("sparse_group_wape", pd.DataFrame())

    print("\n" + "=" * 80)
    print("FINAL JOINED DF CHECK")
    print("=" * 80)

    if len(joined) > 0:
        print("Rows:", len(joined))
        print("ASINs:", joined["asin"].nunique())
        print("Weeks:", joined["order_week"].nunique())
        print("Window:", joined["order_week"].min(), "to", joined["order_week"].max())
        keep_cols = [
            "asin", "order_week", "zero_group", "fbi_demand", "our_price", "true_amt", "pkg_volume", "true_size",
            "p50_amxl", "p70_amxl", "p50_scot", "p70_scot",
        ]
        keep_cols = [c for c in keep_cols if c in joined.columns]
        print(joined[keep_cols].head(20))
    else:
        print("No joined dataframe found.")

    print("\n" + "=" * 80)
    print("SPARSE-GROUP DIAGNOSTIC TABLE")
    print("=" * 80)

    if len(sparse_diag) > 0:
        print(sparse_diag)
    else:
        print("No sparse-group diagnostic table found.")


# =====================================================
# 10. Execute
# =====================================================

# Option A: run model only.
#
# result = run_nb_high_sparse(
#     data_raw1,
#     n_asins=5000,
#     zero_thresholds=(0.4, 0.7),
#     prior_scale=0.3,
#     epochs=60,
#     d_model=32,
#     d_z=16,
#     batch_size=64,
#     M_eval=100,
#     lambda_q=0.05,
#     beta_tail=0.5,
#     patience=5,
#     lambda_z_reg=1.0,
#     remove_extreme=True,
#     extreme_q=0.99,
# )
#
# wape_outputs = run_final_wape(result)


# Option B: run model and WAPE together.
#
# result = run_nb_high_sparse_with_wape(
#     data_raw1,
#     n_asins=5000,
#     zero_thresholds=(0.4, 0.7),
#     prior_scale=0.3,
#     epochs=60,
#     d_model=32,
#     d_z=16,
#     batch_size=64,
#     M_eval=100,
#     lambda_q=0.05,
#     beta_tail=0.5,
#     patience=5,
#     lambda_z_reg=1.0,
#     remove_extreme=True,
#     extreme_q=0.99,
#     remove_oos_dp=True,
# )


# Example:
#
# scot_df = pd.read_csv("scotforecast_2025-12-07_2026-05-10.csv")
#
# result_intersection = run_nb_high_sparse_from_sample_scot_intersection(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     n_asins=5000,
#     seed=42,
#     zero_thresholds=(0.4, 0.7),
#     prior_scale=0.3,
#     epochs=60,
#     history=52,
#     horizon=20,
#     d_model=32,
#     d_z=16,
#     batch_size=64,
#     M_eval=100,
#     lambda_q=0.05,
#     beta_tail=0.5,
#     patience=5,
#     lambda_z_reg=1.0,
#     remove_extreme=True,
#     extreme_q=0.99,
#     run_wape=True,
#     remove_oos_dp=True,
# )


# Main usage: train one model on all sample-SCOT intersection ASINs.
#
# result_all_intersection = run_nb_all_sample_scot_intersection(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     n_asins=5000,
#     seed=42,
#     zero_thresholds=(0.4, 0.7),
#     prior_scale=0.3,
#     epochs=60,
#     history=52,
#     horizon=20,
#     d_model=32,
#     d_z=16,
#     batch_size=64,
#     M_eval=100,
#     lambda_q=0.05,
#     beta_tail=0.5,
#     patience=5,
#     lambda_z_reg=1.0,
#     remove_extreme=True,
#     extreme_q=0.99,
#     run_wape=True,
#     remove_oos_dp=True,
# )
#
# joined_df = result_all_intersection["real_scot_outputs"]["forecast_df_scot_real_with_group"]
# sparse_group_wape = result_all_intersection["real_scot_outputs"]["sparse_group_wape"]
# print(sparse_group_wape)


# Main usage:
#
# result_all_intersection = run_nb_all_sample_scot_intersection(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     n_asins=5000,
#     seed=42,
#     zero_thresholds=(0.4, 0.7),
#     prior_scale=0.3,
#     epochs=60,
#     history=52,
#     horizon=20,
#     d_model=32,
#     d_z=16,
#     batch_size=64,
#     M_eval=100,
#     lambda_q=0.05,
#     beta_tail=0.5,
#     patience=5,
#     lambda_z_reg=1.0,
#     remove_extreme=True,
#     extreme_q=0.99,
#     run_wape=True,
#     remove_oos_dp=True,
# )
#
# print_final_diagnostics(result_all_intersection)
#
# joined_df = result_all_intersection["real_scot_outputs"]["forecast_df_scot_real_with_group"]
# sparse_group_wape = result_all_intersection["real_scot_outputs"]["sparse_group_wape"]


# Main no-distance version with total amount diagnostics:
#
# result_all_intersection = run_nb_all_sample_scot_intersection(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     n_asins=5000,
#     seed=42,
#     zero_thresholds=(0.4, 0.7),
#     prior_scale=0.3,
#     epochs=60,
#     history=52,
#     horizon=20,
#     d_model=32,
#     d_z=16,
#     batch_size=64,
#     M_eval=100,
#     lambda_q=0.05,
#     beta_tail=0.5,
#     patience=5,
#     lambda_z_reg=1.0,
#     remove_extreme=True,
#     extreme_q=0.99,
#     run_wape=True,
#     remove_oos_dp=True,
# )
#
# sparse_group_wape = result_all_intersection["real_scot_outputs"]["sparse_group_wape"]
# joined_df = result_all_intersection["real_scot_outputs"]["forecast_df_scot_real_with_group"]
# print(sparse_group_wape)


# =====================================================
# 15. In-stock feature checker
# =====================================================

def check_instock_feature_setup(result):
    """
    Check the current in_stock_dph setup:
      - history encoder uses raw historical in_stock_dph features
      - future_context excludes in_stock_dph
    """
    data_train = result.get("data_train", None)
    va_ld = result.get("va_ld", None)

    print("\n" + "=" * 80)
    print("IN_STOCK_DPH FEATURE SETUP CHECK")
    print("=" * 80)
    print("History encoder: raw historical in_stock_dph, no shift")
    print("Future context: excludes in_stock_dph")

    if va_ld is not None:
        for batch in va_ld:
            x = batch["x"]
            fc = batch["future_context"]
            print("history x shape:", tuple(x.shape))
            print("future_context shape:", tuple(fc.shape))
            print("history in_stock_dph feature example, first sample:")
            print(x[0, :, 11].detach().cpu().numpy()[:10])
            break

    if data_train is not None:
        print("data_train columns containing stock/instock:")
        print([c for c in data_train.columns if "stock" in c.lower() or "instock" in c.lower()])



# =====================================================
# 16. Context feature checker
# =====================================================

def check_context_feature_columns(data_raw1):
    """
    Print holiday indicator and distance feature columns available in data_raw1.
    """
    holiday_cols = [c for c in data_raw1.columns if c.startswith("holiday_indicator_")]
    distance_cols = [c for c in data_raw1.columns if c.startswith("distance_")]

    print("\n" + "=" * 80)
    print("CONTEXT FEATURE COLUMN CHECK")
    print("=" * 80)
    print("holiday_indicator_* count:", len(holiday_cols))
    print(holiday_cols)
    print("\ndistance_* count:", len(distance_cols))
    print(distance_cols)

    return {
        "holiday_cols": holiday_cols,
        "distance_cols": distance_cols,
    }




# =====================================================
# 16. Stock decoder diagnostics
# =====================================================

def diagnose_stock_decoder(result):
    """
    Diagnose predicted in_stock_dph_hat quality from the integrated stock decoder.
    Uses validation forecast_df generated by the demand model.
    """
    forecast_df = result.get("forecast_df", None)
    if forecast_df is None:
        print("No forecast_df found.")
        return pd.DataFrame()

    required = ["true_future_instock", "pred_instock_dph_hat"]
    missing = [c for c in required if c not in forecast_df.columns]
    if missing:
        print("Missing columns:", missing)
        return pd.DataFrame()

    df = forecast_df.copy()
    y = pd.to_numeric(df["true_future_instock"], errors="coerce").fillna(0).clip(lower=0).values
    p = pd.to_numeric(df["pred_instock_dph_hat"], errors="coerce").fillna(0).clip(lower=0).values

    denom = np.abs(y).sum()
    overall = pd.DataFrame([{
        "rows": len(df),
        "true_mean": y.mean(),
        "pred_mean": p.mean(),
        "WAPE": np.abs(y - p).sum() / denom if denom > 0 else np.nan,
        "log_MAE": np.mean(np.abs(np.log1p(y) - np.log1p(p))),
        "corr": np.corrcoef(y, p)[0, 1] if np.std(y) > 0 and np.std(p) > 0 else np.nan,
        "true_zero_rate": (y <= 0).mean(),
        "pred_zero_rate": (p <= 0).mean(),
    }])

    print("\n" + "=" * 80)
    print("INTEGRATED STOCK DECODER DIAGNOSTIC")
    print("=" * 80)
    print(overall)

    if "horizon" in df.columns:
        rows = []
        for h, g in df.groupby("horizon"):
            yh = pd.to_numeric(g["true_future_instock"], errors="coerce").fillna(0).clip(lower=0).values
            ph = pd.to_numeric(g["pred_instock_dph_hat"], errors="coerce").fillna(0).clip(lower=0).values
            dh = np.abs(yh).sum()
            rows.append({
                "horizon": h,
                "rows": len(g),
                "true_mean": yh.mean(),
                "pred_mean": ph.mean(),
                "WAPE": np.abs(yh - ph).sum() / dh if dh > 0 else np.nan,
                "log_MAE": np.mean(np.abs(np.log1p(yh) - np.log1p(ph))),
                "corr": np.corrcoef(yh, ph)[0, 1] if np.std(yh) > 0 and np.std(ph) > 0 else np.nan,
            })
        by_h = pd.DataFrame(rows)
        print("\nBy horizon:")
        print(by_h)
        return {"overall": overall, "by_horizon": by_h}

    return {"overall": overall}



# Main usage for integrated stock-decoder version:
#
# result_all_intersection = run_nb_all_sample_scot_intersection(
#     data_raw1=data_raw1,
#     scot_df=scot_df,
#     n_asins=5000,
#     seed=42,
#     zero_thresholds=(0.4, 0.7),
#     prior_scale=0.3,
#     epochs=60,
#     history=52,
#     horizon=20,
#     d_model=32,
#     d_z=16,
#     batch_size=64,
#     M_eval=100,
#     lambda_q=0.05,
#     beta_tail=0.5,
#     patience=5,
#     lambda_z_reg=1.0,
#     lambda_stock=0.05,
#     remove_extreme=True,
#     extreme_q=0.99,
#     run_wape=True,
#     remove_oos_dp=True,
# )
#
# diagnose_stock_decoder(result_all_intersection)
# joined_df = result_all_intersection["real_scot_outputs"]["forecast_df_scot_real_with_group"]
# sparse_group_wape = result_all_intersection["real_scot_outputs"]["sparse_group_wape"]



def check_stock_decoder_extra_feature_columns(data_raw1):
    """
    Check which additional stock-decoder features will be used.
    """
    cols = _select_stock_decoder_extra_cols(data_raw1)

    print("\n" + "=" * 80)
    print("STOCK DECODER EXTRA FEATURE COLUMN CHECK")
    print("=" * 80)
    print("count:", len(cols))
    print(cols)

    missing_interesting = [
        c for c in [
            "gl_product_group", "category_code", "brand_class",
            "glance_view_band_cat",
            "hb_rank", "hb_score", "customer_review_count",
            "customer_average_review_rating", "ind_promotion",
            "promotion_amount", "promotion_ratio",
            "pkg_height", "pkg_length", "pkg_width", "pkg_weight",
        ]
        if c not in data_raw1.columns
    ]

    print("\nMissing from recommended list:")
    print(missing_interesting)

    return cols




def check_no_buybox_total_dph_in_context(data_raw1):
    """
    Verify that buy_box_dph and total_dph are not selected as stock decoder extra features.
    """
    cols = _select_stock_decoder_extra_cols(data_raw1)

    bad = [c for c in ["buy_box_dph", "total_dph"] if c in cols]

    print("\n" + "=" * 80)
    print("BUY_BOX / TOTAL_DPH SAFE-CONTEXT CHECK")
    print("=" * 80)
    print("Selected extra feature count:", len(cols))
    print("buy_box_dph or total_dph selected:", bad)

    if len(bad) == 0:
        print("OK: buy_box_dph and total_dph are excluded from stock decoder future context.")
    else:
        print("WARNING: leakage-risk columns are still selected:", bad)

    return cols




def check_safe_historical_dph_proxy_context(result, n_batches=1):
    """
    Check that historical DPH proxy columns are present and constant within horizon.
    """
    va_ld = result.get("va_ld", None)
    context_cols = result.get("context_cols", None)

    print("\n" + "=" * 80)
    print("SAFE HISTORICAL DPH PROXY CONTEXT CHECK")
    print("=" * 80)

    if context_cols is not None:
        proxy_cols = [c for c in context_cols if c.startswith("hist_total_dph") or c.startswith("hist_buy_box_dph")]
        print("Proxy cols:", proxy_cols)

    if va_ld is None:
        print("No va_ld found.")
        return

    for bi, b in enumerate(va_ld):
        fc = b["future_context"].detach().cpu().numpy()
        print("future_context shape:", fc.shape)

        if context_cols is not None:
            for c in proxy_cols:
                j = context_cols.index(c)
                print(c, "first sample values:", fc[0, :, j])
                print(c, "unique first sample:", np.unique(fc[0, :, j]))

        if bi + 1 >= n_batches:
            break

