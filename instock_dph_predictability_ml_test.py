
"""
Standalone in_stock_dph forecasting diagnostic.

Goal:
    Test whether future in_stock_dph can be predicted from forecast-origin-safe features.

This script does NOT modify the demand model.
It only checks whether in_stock_dph is predictable enough to be used later as a predicted future covariate.

Main usage:

    instock_table = build_instock_forecast_table(
        data_raw1=data_raw1,
        n_asins=5000,
        seed=42,
        history=52,
        horizon=20,
        target_col="in_stock_dph",
    )

    instock_outputs = evaluate_instock_forecaster(
        instock_df=instock_table,
        valid_start=None,
    )

    print(instock_outputs["summary"])

"""

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# 1. Utility
# =====================================================

def _safe_num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _corr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return np.nan

    if np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _wape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denom = np.abs(y_true).sum()

    if denom <= 1e-12:
        return np.nan

    return float(np.abs(y_true - y_pred).sum() / denom)


def _safe_week_of_year(dt):
    return int(pd.Timestamp(dt).isocalendar().week)


def _make_zero_streak(arr):
    streak = 0
    for v in arr[::-1]:
        if v <= 0:
            streak += 1
        else:
            break
    return streak


# =====================================================
# 2. Build forecast-origin-safe table
# =====================================================

def build_instock_forecast_table(
    data_raw1,
    n_asins=5000,
    seed=42,
    history=52,
    horizon=20,
    target_col="in_stock_dph",
    demand_col="fbi_demand",
    price_col="our_price",
    asin_col="asin",
    week_col="order_week",
    use_scot_intersection=False,
    scot_df=None,
):
    """
    Build a forecast-origin table for predicting future in_stock_dph.

    Each row is one:
        ASIN + forecast origin + target horizon

    Target:
        target_log_instock = log1p(in_stock_dph at target week)

    All lag / rolling features are computed from history ending at forecast origin.
    Future-known calendar features are taken from the target week.

    Parameters
    ----------
    use_scot_intersection:
        If True, sample ASINs first, then keep only ASINs also present in scot_df.
        This matches the demand-model comparison universe more closely.
    """
    df = data_raw1.copy()
    df.columns = [c.strip() for c in df.columns]

    required_cols = [asin_col, week_col, target_col, demand_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df[asin_col] = df[asin_col].astype(str)
    df[week_col] = pd.to_datetime(df[week_col])

    rng = np.random.default_rng(seed)
    all_asins = df[asin_col].dropna().unique()

    sample_asins = rng.choice(
        all_asins,
        size=min(n_asins, len(all_asins)),
        replace=False,
    )

    sample_asin_set = set(sample_asins)

    if use_scot_intersection:
        if scot_df is None:
            raise ValueError("scot_df must be provided when use_scot_intersection=True.")

        scot = scot_df.copy()
        scot.columns = [c.strip() for c in scot.columns]

        if asin_col not in scot.columns:
            raise ValueError(f"SCOT df is missing ASIN column: {asin_col}")

        scot[asin_col] = scot[asin_col].astype(str)
        scot_asin_set = set(scot[asin_col].dropna().unique())

        keep_asins = sorted(sample_asin_set & scot_asin_set)

        print("=" * 80)
        print("SAMPLE-SCOT ASIN INTERSECTION FOR IN_STOCK_DPH TEST")
        print("=" * 80)
        print("Sample ASINs:", len(sample_asin_set))
        print("SCOT ASINs:", len(scot_asin_set))
        print("Intersection ASINs:", len(keep_asins))
    else:
        keep_asins = sorted(sample_asin_set)

    df = df[df[asin_col].isin(keep_asins)].copy()

    df[target_col] = _safe_num(df[target_col], 0.0).clip(lower=0)
    df[demand_col] = _safe_num(df[demand_col], 0.0).clip(lower=0)

    if price_col in df.columns:
        df[price_col] = _safe_num(df[price_col], 0.0).clip(lower=0)
        df["log_price"] = np.log1p(df[price_col])
    else:
        df["log_price"] = 0.0

    holiday_cols = [c for c in df.columns if c.startswith("holiday_indicator_")]
    distance_cols = [c for c in df.columns if c.startswith("distance_")]

    for c in holiday_cols:
        df[c] = _safe_num(df[c], 0.0).clip(lower=0, upper=1)

    for c in distance_cols:
        df[c] = _safe_num(df[c], 0.0).clip(lower=-12, upper=12) / 12.0

    rows = []

    for asin, g in df.groupby(asin_col):
        g = g.sort_values(week_col).reset_index(drop=True)

        if len(g) < history + horizon:
            continue

        instock = g[target_col].values.astype(float)
        demand = g[demand_col].values.astype(float)

        log_instock = np.log1p(instock)
        log_demand = np.log1p(demand)

        for start in range(0, len(g) - history - horizon + 1):
            origin_idx = start + history - 1

            hist_instock = instock[start:start + history]
            hist_log_instock = log_instock[start:start + history]
            hist_demand = demand[start:start + history]
            hist_log_demand = log_demand[start:start + history]

            # Forecast-origin-safe history features.
            instock_last = hist_instock[-1]
            instock_log_last = hist_log_instock[-1]
            instock_mean_4 = hist_instock[-4:].mean()
            instock_mean_13 = hist_instock[-13:].mean()
            instock_mean_26 = hist_instock[-26:].mean() if len(hist_instock) >= 26 else hist_instock.mean()
            instock_mean_52 = hist_instock.mean()

            instock_log_mean_4 = hist_log_instock[-4:].mean()
            instock_log_mean_13 = hist_log_instock[-13:].mean()
            instock_log_mean_26 = hist_log_instock[-26:].mean() if len(hist_log_instock) >= 26 else hist_log_instock.mean()
            instock_log_mean_52 = hist_log_instock.mean()

            instock_std_13 = hist_instock[-13:].std()
            instock_zero_rate_13 = (hist_instock[-13:] <= 0).mean()
            instock_zero_streak = _make_zero_streak(hist_instock)

            instock_trend_4 = hist_log_instock[-1] - hist_log_instock[-4] if history >= 4 else 0.0
            instock_trend_13 = hist_log_instock[-1] - hist_log_instock[-13] if history >= 13 else 0.0

            demand_last = hist_demand[-1]
            demand_log_last = hist_log_demand[-1]
            demand_mean_4 = hist_demand[-4:].mean()
            demand_mean_13 = hist_demand[-13:].mean()
            demand_mean_26 = hist_demand[-26:].mean() if len(hist_demand) >= 26 else hist_demand.mean()
            demand_active_rate_13 = (hist_demand[-13:] > 0).mean()
            demand_zero_streak = _make_zero_streak(hist_demand)
            demand_recent_peak_13 = hist_demand[-13:].max()

            origin_week = g.loc[origin_idx, week_col]

            for h in range(1, horizon + 1):
                target_idx = origin_idx + h
                target_week = g.loc[target_idx, week_col]
                week_of_year = _safe_week_of_year(target_week)

                row = {
                    asin_col: asin,
                    "origin_week": origin_week,
                    "target_week": target_week,
                    "horizon": h,

                    # In-stock history features.
                    "instock_last": instock_last,
                    "instock_log_last": instock_log_last,
                    "instock_mean_4": instock_mean_4,
                    "instock_mean_13": instock_mean_13,
                    "instock_mean_26": instock_mean_26,
                    "instock_mean_52": instock_mean_52,
                    "instock_log_mean_4": instock_log_mean_4,
                    "instock_log_mean_13": instock_log_mean_13,
                    "instock_log_mean_26": instock_log_mean_26,
                    "instock_log_mean_52": instock_log_mean_52,
                    "instock_std_13": instock_std_13,
                    "instock_zero_rate_13": instock_zero_rate_13,
                    "instock_zero_streak": instock_zero_streak,
                    "instock_trend_4": instock_trend_4,
                    "instock_trend_13": instock_trend_13,

                    # Demand history features.
                    "demand_last": demand_last,
                    "demand_log_last": demand_log_last,
                    "demand_mean_4": demand_mean_4,
                    "demand_mean_13": demand_mean_13,
                    "demand_mean_26": demand_mean_26,
                    "demand_active_rate_13": demand_active_rate_13,
                    "demand_zero_streak": demand_zero_streak,
                    "demand_recent_peak_13": demand_recent_peak_13,

                    # Future-known features.
                    "log_price": g.loc[target_idx, "log_price"],
                    "week_sin": np.sin(2 * np.pi * week_of_year / 52),
                    "week_cos": np.cos(2 * np.pi * week_of_year / 52),

                    # Target.
                    "target_instock": instock[target_idx],
                    "target_log_instock": np.log1p(instock[target_idx]),

                    # Simple baselines.
                    "baseline_last": instock_last,
                    "baseline_mean_4": instock_mean_4,
                    "baseline_mean_13": instock_mean_13,
                    "baseline_mean_26": instock_mean_26,
                    "baseline_log_mean_4": np.expm1(instock_log_mean_4),
                    "baseline_log_mean_13": np.expm1(instock_log_mean_13),
                }

                for c in holiday_cols:
                    row[c] = g.loc[target_idx, c]

                for c in distance_cols:
                    row[c] = g.loc[target_idx, c]

                rows.append(row)

    out = pd.DataFrame(rows)

    print("=" * 80)
    print("IN_STOCK_DPH FORECAST TABLE")
    print("=" * 80)
    print("Rows:", len(out))
    print("ASINs:", out[asin_col].nunique() if len(out) > 0 else 0)

    if len(out) > 0:
        print("Origin weeks:", out["origin_week"].min(), "to", out["origin_week"].max())
        print("Target weeks:", out["target_week"].min(), "to", out["target_week"].max())
        print("Target mean:", out["target_instock"].mean())
        print("Target zero rate:", (out["target_instock"] <= 0).mean())

    print("Holiday cols:", len(holiday_cols))
    print("Distance cols:", len(distance_cols))

    return out


# =====================================================
# 3. Evaluation
# =====================================================

def _summarize_prediction(valid_df, pred_col):
    y_true = valid_df["target_instock"].values.astype(float)
    y_pred = valid_df[pred_col].values.astype(float)
    y_pred = np.clip(y_pred, 0, None)

    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)

    return {
        "model": pred_col,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": _rmse(y_true, y_pred),
        "WAPE": _wape(y_true, y_pred),
        "log_MAE": mean_absolute_error(y_true_log, y_pred_log),
        "log_RMSE": _rmse(y_true_log, y_pred_log),
        "R2_log": r2_score(y_true_log, y_pred_log),
        "corr_raw": _corr(y_true, y_pred),
        "true_mean": y_true.mean(),
        "pred_mean": y_pred.mean(),
        "true_zero_rate": (y_true <= 0).mean(),
        "pred_zero_rate": (y_pred <= 0).mean(),
    }


def evaluate_instock_forecaster(
    instock_df,
    valid_start=None,
    valid_frac_by_time=0.2,
    model_type="hgb",
    random_state=42,
):
    """
    Train a simple ML model to predict log1p(in_stock_dph).

    Splitting:
        time-based split by target_week.
        If valid_start is None, use the last valid_frac_by_time target weeks as validation.

    model_type:
        "hgb"    = HistGradientBoostingRegressor
        "ridge"  = Ridge
        "rf"     = RandomForestRegressor

    Returns:
        model, feature_cols, valid_pred_df, summary
    """
    df = instock_df.copy()
    df["target_week"] = pd.to_datetime(df["target_week"])

    if len(df) == 0:
        raise ValueError("instock_df is empty.")

    weeks = np.array(sorted(df["target_week"].dropna().unique()))

    if valid_start is None:
        split_idx = int(np.floor(len(weeks) * (1 - valid_frac_by_time)))
        split_idx = max(1, min(split_idx, len(weeks) - 1))
        valid_start = weeks[split_idx]

    train_df = df[df["target_week"] < valid_start].copy()
    valid_df = df[df["target_week"] >= valid_start].copy()

    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("Train or validation split is empty. Check valid_start.")

    drop_cols = [
        "asin",
        "origin_week",
        "target_week",
        "target_instock",
        "target_log_instock",
        "baseline_last",
        "baseline_mean_4",
        "baseline_mean_13",
        "baseline_mean_26",
        "baseline_log_mean_4",
        "baseline_log_mean_13",
    ]

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["target_log_instock"].values.astype(float)

    X_valid = valid_df[feature_cols].fillna(0.0)

    if model_type == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=random_state,
        )
    elif model_type == "ridge":
        model = Ridge(alpha=1.0, random_state=random_state)
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError("model_type must be one of: hgb, ridge, rf")

    model.fit(X_train, y_train)

    pred_log = model.predict(X_valid)
    valid_df["pred_instock_ml"] = np.expm1(pred_log).clip(min=0)

    for c in [
        "baseline_last",
        "baseline_mean_4",
        "baseline_mean_13",
        "baseline_mean_26",
        "baseline_log_mean_4",
        "baseline_log_mean_13",
    ]:
        valid_df[c] = valid_df[c].clip(lower=0)

    summary = pd.DataFrame([
        _summarize_prediction(valid_df, "baseline_last"),
        _summarize_prediction(valid_df, "baseline_mean_4"),
        _summarize_prediction(valid_df, "baseline_mean_13"),
        _summarize_prediction(valid_df, "baseline_mean_26"),
        _summarize_prediction(valid_df, "baseline_log_mean_4"),
        _summarize_prediction(valid_df, "baseline_log_mean_13"),
        _summarize_prediction(valid_df, "pred_instock_ml"),
    ])

    print("\n" + "=" * 80)
    print("IN_STOCK_DPH FORECAST EVALUATION")
    print("=" * 80)
    print("Model type:", model_type)
    print("Train rows:", len(train_df))
    print("Valid rows:", len(valid_df))
    print("Train target weeks:", train_df["target_week"].min(), "to", train_df["target_week"].max())
    print("Valid target weeks:", valid_df["target_week"].min(), "to", valid_df["target_week"].max())
    print("Valid start:", valid_start)
    print("Feature count:", len(feature_cols))
    print("\nSummary:")
    print(summary)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "train_df": train_df,
        "valid_pred_df": valid_df,
        "summary": summary,
        "valid_start": valid_start,
    }


# =====================================================
# 4. Horizon-level diagnosis
# =====================================================

def summarize_instock_by_horizon(valid_pred_df):
    """
    Evaluate baselines and ML forecast by horizon.
    """
    rows = []

    for h, g in valid_pred_df.groupby("horizon"):
        for pred_col in [
            "baseline_last",
            "baseline_mean_4",
            "baseline_mean_13",
            "pred_instock_ml",
        ]:
            s = _summarize_prediction(g, pred_col)
            s["horizon"] = h
            rows.append(s)

    out = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("IN_STOCK_DPH FORECAST BY HORIZON")
    print("=" * 80)
    print(out[[
        "horizon",
        "model",
        "WAPE",
        "log_MAE",
        "corr_raw",
        "true_mean",
        "pred_mean",
    ]])

    return out


# =====================================================
# 5. Example helper
# =====================================================

def run_instock_predictability_test(
    data_raw1,
    n_asins=5000,
    seed=42,
    history=52,
    horizon=20,
    target_col="in_stock_dph",
    model_type="hgb",
    use_scot_intersection=False,
    scot_df=None,
):
    """
    End-to-end predictability test for in_stock_dph.
    """
    instock_table = build_instock_forecast_table(
        data_raw1=data_raw1,
        n_asins=n_asins,
        seed=seed,
        history=history,
        horizon=horizon,
        target_col=target_col,
        use_scot_intersection=use_scot_intersection,
        scot_df=scot_df,
    )

    outputs = evaluate_instock_forecaster(
        instock_df=instock_table,
        valid_start=None,
        valid_frac_by_time=0.2,
        model_type=model_type,
        random_state=42,
    )

    horizon_summary = summarize_instock_by_horizon(
        outputs["valid_pred_df"]
    )

    outputs["instock_table"] = instock_table
    outputs["horizon_summary"] = horizon_summary

    return outputs


# Example:
#
# instock_outputs = run_instock_predictability_test(
#     data_raw1=data_raw1,
#     n_asins=5000,
#     seed=42,
#     history=52,
#     horizon=20,
#     target_col="in_stock_dph",
#     model_type="hgb",
#     use_scot_intersection=False,
#     scot_df=None,
# )
#
# print(instock_outputs["summary"])
# print(instock_outputs["horizon_summary"])
