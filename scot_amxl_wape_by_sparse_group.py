
import pandas as pd
import numpy as np


def build_sample_asin_and_sparse_groups(
    data_raw1,
    n_asins=5000,
    seed=42,
    zero_thresholds=(0.4, 0.7),
):
    df = data_raw1.copy()
    df.columns = [c.strip() for c in df.columns]

    for c in ["asin", "order_week", "fbi_demand"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column in data_raw1: {c}")

    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])
    df["fbi_demand"] = pd.to_numeric(
        df["fbi_demand"],
        errors="coerce"
    ).fillna(0).clip(lower=0)

    rng = np.random.default_rng(seed)
    unique_asins = df["asin"].dropna().unique()

    sample_asins = rng.choice(
        unique_asins,
        size=min(n_asins, len(unique_asins)),
        replace=False
    )

    sample_asin_df = pd.DataFrame({"asin": sample_asins})
    sample_df = df[df["asin"].isin(sample_asins)].copy()

    asin_stats = (
        sample_df
        .groupby("asin", as_index=False)
        .agg(
            zero_rate=("fbi_demand", lambda x: (x == 0).mean()),
            total_demand=("fbi_demand", "sum"),
            n_weeks=("fbi_demand", "count"),
        )
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

    sample_asin_group_df = sample_asin_df.merge(
        asin_stats[["asin", "zero_rate", "zero_group", "total_demand", "n_weeks"]],
        on="asin",
        how="left"
    )

    print("=" * 80)
    print("SAMPLED ASIN GROUP SUMMARY")
    print("=" * 80)
    print("Sample ASINs:", sample_asin_group_df["asin"].nunique())
    print(sample_asin_group_df.groupby("zero_group", dropna=False)["asin"].nunique())

    return sample_asin_group_df, sample_df, asin_stats


def get_true_demand_for_eval(data_raw1):
    df = data_raw1.copy()
    df.columns = [c.strip() for c in df.columns]

    for c in ["asin", "order_week", "fbi_demand"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column in data_raw1: {c}")

    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])
    df["fbi_demand"] = pd.to_numeric(
        df["fbi_demand"],
        errors="coerce"
    ).fillna(0).clip(lower=0)

    keep_cols = ["asin", "order_week", "fbi_demand"]

    for c in ["scot_oos", "oos_status", "oos"]:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    true_df = (
        df[keep_cols]
        .drop_duplicates(subset=["asin", "order_week"])
        .reset_index(drop=True)
    )

    print("\nTRUE DEMAND DF")
    print("rows:", len(true_df))
    print("unique ASINs:", true_df["asin"].nunique())

    return true_df


def standardize_forecast_df(
    forecast_df,
    source="scot",
    p50_col=None,
    p70_col=None,
):
    df = forecast_df.copy()
    df.columns = [c.strip() for c in df.columns]

    for c in ["asin", "order_week"]:
        if c not in df.columns:
            raise ValueError(f"Missing required forecast column: {c}")

    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])

    if p50_col is None:
        p50_col = "forecast_qty_p50" if source == "scot" else "p50_amxl"

    if p70_col is None:
        p70_col = "forecast_qty_p70" if source == "scot" else "p70_amxl"

    if p50_col not in df.columns:
        raise ValueError(f"{p50_col} not found in forecast_df.")
    if p70_col not in df.columns:
        raise ValueError(f"{p70_col} not found in forecast_df.")

    df["p50_pred"] = pd.to_numeric(df[p50_col], errors="coerce")
    df["p70_pred"] = pd.to_numeric(df[p70_col], errors="coerce")

    keep_cols = ["asin", "order_week", "p50_pred", "p70_pred"]

    optional_cols = [
        "forecast_qty_p50",
        "forecast_qty_p70",
        "forecast_qty_p90",
        "p50_amxl",
        "p70_amxl",
        "model_id",
        "fcst_start_week",
        "fcst_week_index",
    ]

    for c in optional_cols:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    out = df[keep_cols].copy()

    if "fcst_start_week" in out.columns:
        out["fcst_start_week"] = pd.to_datetime(out["fcst_start_week"])

    return out


def build_eval_df_for_forecast(
    forecast_df,
    data_raw1,
    sample_asin_group_df,
    source="scot",
    p50_col=None,
    p70_col=None,
):
    pred_df = standardize_forecast_df(
        forecast_df,
        source=source,
        p50_col=p50_col,
        p70_col=p70_col,
    )

    true_df = get_true_demand_for_eval(data_raw1)

    sample_group = sample_asin_group_df.copy()
    sample_group["asin"] = sample_group["asin"].astype(str)

    pred_before = pred_df.copy()

    pred_df = pred_df.merge(
        sample_group[["asin", "zero_rate", "zero_group"]],
        on="asin",
        how="inner"
    )

    eval_df = pred_df.merge(
        true_df,
        on=["asin", "order_week"],
        how="inner"
    )

    print("\n" + "=" * 80)
    print("BUILD EVAL DF")
    print("=" * 80)
    print("source:", source)
    print("forecast rows before sample ASIN filter:", len(pred_before))
    print("forecast ASINs before sample ASIN filter:", pred_before["asin"].nunique())
    print("rows after sample ASIN filter + true demand merge:", len(eval_df))
    print("unique ASINs after merge:", eval_df["asin"].nunique())
    print(eval_df.groupby("zero_group", dropna=False)["asin"].nunique())

    return eval_df


def add_boss_quantile_penalty(
    df,
    actual_col="fbi_demand",
    pred_col="p50_pred",
    prefix="p50",
    quantile=0.5,
):
    out = df.copy()

    y = pd.to_numeric(out[actual_col], errors="coerce").fillna(0)
    pred = pd.to_numeric(out[pred_col], errors="coerce").fillna(0)

    over_col = f"{prefix}_overbias"
    under_col = f"{prefix}_underbias"
    penalty_col = f"{prefix}_penalty"
    row_wape_col = f"{prefix}_row_wape"

    out[over_col] = np.where(
        pred >= y,
        np.abs(pred - y) * (1.0 - quantile),
        0.0
    )

    out[under_col] = np.where(
        pred < y,
        np.abs(pred - y) * quantile,
        0.0
    )

    out[penalty_col] = out[over_col] + out[under_col]

    out[row_wape_col] = np.where(
        y != 0,
        out[penalty_col] / y,
        np.nan
    )

    return out


def _infer_oos_col(df, oos_col=None):
    if oos_col is not None and oos_col in df.columns:
        return oos_col

    for c in ["oos_status", "scot_oos", "oos"]:
        if c in df.columns:
            return c

    return None


def calculate_group_wape_summary(
    eval_df,
    remove_oos_dp=True,
    oos_col=None,
    label="forecast",
):
    df = eval_df.copy()

    oos_col = _infer_oos_col(df, oos_col=oos_col)

    print("\n" + "=" * 80)
    print(f"WAPE CALCULATION: {label}")
    print("=" * 80)
    print("Before OOS filtering:", df.shape)

    if remove_oos_dp and oos_col is not None and oos_col in df.columns:
        df[oos_col] = pd.to_numeric(df[oos_col], errors="coerce").fillna(0)
        df = df[df[oos_col] == 0].copy()
        print(f"After OOS filtering by {oos_col}:", df.shape)
    else:
        print("OOS filtering skipped.")

    df = add_boss_quantile_penalty(
        df,
        actual_col="fbi_demand",
        pred_col="p50_pred",
        prefix="p50",
        quantile=0.5,
    )

    df = add_boss_quantile_penalty(
        df,
        actual_col="fbi_demand",
        pred_col="p70_pred",
        prefix="p70",
        quantile=0.7,
    )

    def summarize_one_group(g, group_name):
        denom = g["fbi_demand"].sum()

        if denom == 0:
            p50_penalty = np.nan
            p70_penalty = np.nan
            p50_under = np.nan
            p50_over = np.nan
            p70_under = np.nan
            p70_over = np.nan
        else:
            p50_penalty = g["p50_penalty"].sum() / denom
            p50_under = g["p50_underbias"].sum() / denom
            p50_over = g["p50_overbias"].sum() / denom

            p70_penalty = g["p70_penalty"].sum() / denom
            p70_under = g["p70_underbias"].sum() / denom
            p70_over = g["p70_overbias"].sum() / denom

        return {
            "label": label,
            "zero_group": group_name,
            "n_rows": len(g),
            "n_asins": g["asin"].nunique(),
            "total_fbi_demand": denom,
            "true_mean": g["fbi_demand"].mean(),
            "p50_mean": g["p50_pred"].mean(),
            "p70_mean": g["p70_pred"].mean(),
            "true_zero_rate": (g["fbi_demand"] == 0).mean(),
            "true_active_ratio": (g["fbi_demand"] > 0).mean(),
            "p50_active_ratio": (g["p50_pred"] > 0.5).mean(),
            "p70_active_ratio": (g["p70_pred"] > 0.5).mean(),
            "p50_penalty": p50_penalty,
            "p50_underbias": p50_under,
            "p50_overbias": p50_over,
            "p70_penalty": p70_penalty,
            "p70_underbias": p70_under,
            "p70_overbias": p70_over,
        }

    rows = []

    for group_name, g in df.groupby("zero_group", dropna=False):
        rows.append(summarize_one_group(g, group_name))

    rows.append(summarize_one_group(df, "ALL"))

    summary_df = pd.DataFrame(rows)

    print("\nWAPE summary:")
    display_cols = [
        "label",
        "zero_group",
        "n_rows",
        "n_asins",
        "true_zero_rate",
        "true_mean",
        "p50_mean",
        "p70_mean",
        "p50_penalty",
        "p70_penalty",
        "p50_underbias",
        "p50_overbias",
        "p70_underbias",
        "p70_overbias",
    ]
    existing = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[existing])

    return df, summary_df


def run_scot_wape_by_sparse_group(
    scot_df_or_path,
    data_raw1,
    n_asins=5000,
    seed=42,
    zero_thresholds=(0.4, 0.7),
    remove_oos_dp=True,
    oos_col=None,
):
    if isinstance(scot_df_or_path, str):
        scot_df = pd.read_csv(scot_df_or_path)
    else:
        scot_df = scot_df_or_path.copy()

    sample_asin_group_df, sample_df, asin_stats = build_sample_asin_and_sparse_groups(
        data_raw1=data_raw1,
        n_asins=n_asins,
        seed=seed,
        zero_thresholds=zero_thresholds,
    )

    eval_df = build_eval_df_for_forecast(
        forecast_df=scot_df,
        data_raw1=data_raw1,
        sample_asin_group_df=sample_asin_group_df,
        source="scot",
        p50_col="forecast_qty_p50",
        p70_col="forecast_qty_p70",
    )

    detail_df, summary_df = calculate_group_wape_summary(
        eval_df,
        remove_oos_dp=remove_oos_dp,
        oos_col=oos_col,
        label="SCOT",
    )

    return {
        "sample_asin_group_df": sample_asin_group_df,
        "sample_df": sample_df,
        "asin_stats": asin_stats,
        "eval_df": eval_df,
        "detail_df": detail_df,
        "summary_df": summary_df,
    }


def run_amxl_wape_by_sparse_group(
    forecast_df,
    data_raw1,
    n_asins=5000,
    seed=42,
    zero_thresholds=(0.4, 0.7),
    remove_oos_dp=True,
    oos_col=None,
):
    sample_asin_group_df, sample_df, asin_stats = build_sample_asin_and_sparse_groups(
        data_raw1=data_raw1,
        n_asins=n_asins,
        seed=seed,
        zero_thresholds=zero_thresholds,
    )

    eval_df = build_eval_df_for_forecast(
        forecast_df=forecast_df,
        data_raw1=data_raw1,
        sample_asin_group_df=sample_asin_group_df,
        source="amxl",
        p50_col="p50_amxl",
        p70_col="p70_amxl",
    )

    detail_df, summary_df = calculate_group_wape_summary(
        eval_df,
        remove_oos_dp=remove_oos_dp,
        oos_col=oos_col,
        label="AMXL",
    )

    return {
        "sample_asin_group_df": sample_asin_group_df,
        "sample_df": sample_df,
        "asin_stats": asin_stats,
        "eval_df": eval_df,
        "detail_df": detail_df,
        "summary_df": summary_df,
    }


def compare_scot_amxl_by_sparse_group(
    scot_df_or_path,
    forecast_df,
    data_raw1,
    n_asins=5000,
    seed=42,
    zero_thresholds=(0.4, 0.7),
    remove_oos_dp=True,
    oos_col=None,
):
    if isinstance(scot_df_or_path, str):
        scot_df = pd.read_csv(scot_df_or_path)
    else:
        scot_df = scot_df_or_path.copy()

    sample_asin_group_df, sample_df, asin_stats = build_sample_asin_and_sparse_groups(
        data_raw1=data_raw1,
        n_asins=n_asins,
        seed=seed,
        zero_thresholds=zero_thresholds,
    )

    scot_eval_df = build_eval_df_for_forecast(
        forecast_df=scot_df,
        data_raw1=data_raw1,
        sample_asin_group_df=sample_asin_group_df,
        source="scot",
        p50_col="forecast_qty_p50",
        p70_col="forecast_qty_p70",
    )

    scot_detail_df, scot_summary_df = calculate_group_wape_summary(
        scot_eval_df,
        remove_oos_dp=remove_oos_dp,
        oos_col=oos_col,
        label="SCOT",
    )

    amxl_eval_df = build_eval_df_for_forecast(
        forecast_df=forecast_df,
        data_raw1=data_raw1,
        sample_asin_group_df=sample_asin_group_df,
        source="amxl",
        p50_col="p50_amxl",
        p70_col="p70_amxl",
    )

    amxl_detail_df, amxl_summary_df = calculate_group_wape_summary(
        amxl_eval_df,
        remove_oos_dp=remove_oos_dp,
        oos_col=oos_col,
        label="AMXL",
    )

    combined_summary = pd.concat(
        [scot_summary_df, amxl_summary_df],
        ignore_index=True
    )

    metric_cols = [
        "p50_penalty",
        "p70_penalty",
        "p50_underbias",
        "p50_overbias",
        "p70_underbias",
        "p70_overbias",
        "true_zero_rate",
        "true_mean",
        "p50_mean",
        "p70_mean",
    ]

    pivot_summary = combined_summary.pivot_table(
        index="zero_group",
        columns="label",
        values=[c for c in metric_cols if c in combined_summary.columns],
        aggfunc="first"
    )

    print("\n" + "=" * 80)
    print("COMBINED SCOT VS AMXL SUMMARY")
    print("=" * 80)
    print(combined_summary)

    print("\nPivot summary:")
    print(pivot_summary)

    return {
        "sample_asin_group_df": sample_asin_group_df,
        "sample_df": sample_df,
        "asin_stats": asin_stats,
        "scot_eval_df": scot_eval_df,
        "scot_detail_df": scot_detail_df,
        "scot_summary_df": scot_summary_df,
        "amxl_eval_df": amxl_eval_df,
        "amxl_detail_df": amxl_detail_df,
        "amxl_summary_df": amxl_summary_df,
        "combined_summary": combined_summary,
        "pivot_summary": pivot_summary,
    }


"""
Example 1: SCOT only

scot_outputs = run_scot_wape_by_sparse_group(
    scot_df_or_path="scotforecast_2025-12-07_2026-04-19.csv",
    data_raw1=data_raw1,
    n_asins=5000,
    seed=42,
    zero_thresholds=(0.4, 0.7),
    remove_oos_dp=True,
)

scot_summary_df = scot_outputs["summary_df"]
print(scot_summary_df)


Example 2: AMXL only

amxl_outputs = run_amxl_wape_by_sparse_group(
    forecast_df=forecast_df,
    data_raw1=data_raw1,
    n_asins=5000,
    seed=42,
    zero_thresholds=(0.4, 0.7),
    remove_oos_dp=True,
)

amxl_summary_df = amxl_outputs["summary_df"]
print(amxl_summary_df)


Example 3: SCOT vs AMXL together

comparison_outputs = compare_scot_amxl_by_sparse_group(
    scot_df_or_path="scotforecast_2025-12-07_2026-04-19.csv",
    forecast_df=forecast_df,
    data_raw1=data_raw1,
    n_asins=5000,
    seed=42,
    zero_thresholds=(0.4, 0.7),
    remove_oos_dp=True,
)

combined_summary = comparison_outputs["combined_summary"]
pivot_summary = comparison_outputs["pivot_summary"]

print(combined_summary)
print(pivot_summary)
"""
