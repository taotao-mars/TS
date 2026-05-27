import pandas as pd
import numpy as np


def run_high_sparse_scot_alignment_wape(
    result,
    scot_df,
    remove_oos_dp=True,
    source="lp",
):
    """
    Align real SCOT forecasts to your high-sparse NB forecast_df
    and compute boss-style WAPE using existing WAPE functions.

    Required:
      result["forecast_df"] has:
        asin, order_week, fbi_demand, p50_amxl, p70_amxl, oos_status/oos/scot_oos

      scot_df has:
        asin, order_week, forecast_qty_p50, forecast_qty_p70
    """

    if "calculate_wape_using_lp_oos2" not in globals():
        raise RuntimeError("calculate_wape_using_lp_oos2 is not defined.")

    if "quick_error_check" not in globals():
        raise RuntimeError("quick_error_check is not defined.")

    # -----------------------------
    # 1. Prepare NB high-sparse forecast
    # -----------------------------
    forecast_df = result["forecast_df"].copy()
    forecast_df.columns = [c.strip() for c in forecast_df.columns]

    forecast_df["asin"] = forecast_df["asin"].astype(str)
    forecast_df["order_week"] = pd.to_datetime(forecast_df["order_week"])

    print("=" * 80)
    print("NB HIGH-SPARSE FORECAST WINDOW")
    print("=" * 80)
    print("NB rows:", len(forecast_df))
    print("NB ASINs:", forecast_df["asin"].nunique())
    print("NB weeks:", forecast_df["order_week"].min(), "to", forecast_df["order_week"].max())
    print("NB week count:", forecast_df["order_week"].nunique())

    # -----------------------------
    # 2. Prepare real SCOT forecast
    # -----------------------------
    scot = scot_df.copy()
    scot.columns = [c.strip() for c in scot.columns]

    required_scot_cols = [
        "asin",
        "order_week",
        "forecast_qty_p50",
        "forecast_qty_p70",
    ]

    for c in required_scot_cols:
        if c not in scot.columns:
            raise ValueError(f"Missing SCOT column: {c}")

    scot["asin"] = scot["asin"].astype(str)
    scot["order_week"] = pd.to_datetime(scot["order_week"])

    scot["forecast_qty_p50"] = pd.to_numeric(
        scot["forecast_qty_p50"],
        errors="coerce"
    )

    scot["forecast_qty_p70"] = pd.to_numeric(
        scot["forecast_qty_p70"],
        errors="coerce"
    )

    if "fcst_start_week" in scot.columns:
        scot["fcst_start_week"] = pd.to_datetime(scot["fcst_start_week"])

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

    # -----------------------------
    # 3. Align by asin + order_week
    # -----------------------------
    scot_keep = scot[
        [
            "asin",
            "order_week",
            "forecast_qty_p50",
            "forecast_qty_p70",
        ]
    ].copy()

    # In case SCOT has duplicate rows for the same ASIN-week
    scot_keep = (
        scot_keep
        .groupby(["asin", "order_week"], as_index=False)
        .agg(
            forecast_qty_p50=("forecast_qty_p50", "mean"),
            forecast_qty_p70=("forecast_qty_p70", "mean"),
        )
    )

    forecast_df_scot_real = forecast_df.merge(
        scot_keep,
        on=["asin", "order_week"],
        how="inner"
    )

    print("\n" + "=" * 80)
    print("ALIGNMENT CHECK")
    print("=" * 80)
    print("NB forecast rows:", len(forecast_df))
    print("After SCOT merge rows:", len(forecast_df_scot_real))
    print("Matched ASINs:", forecast_df_scot_real["asin"].nunique())
    print(
        "Matched weeks:",
        forecast_df_scot_real["order_week"].min(),
        "to",
        forecast_df_scot_real["order_week"].max()
    )
    print("Matched week count:", forecast_df_scot_real["order_week"].nunique())

    row_match_rate = len(forecast_df_scot_real) / max(len(forecast_df), 1)
    asin_match_rate = (
        forecast_df_scot_real["asin"].nunique()
        / max(forecast_df["asin"].nunique(), 1)
    )

    print("Row match rate:", row_match_rate)
    print("ASIN match rate:", asin_match_rate)

    if row_match_rate < 0.8:
        print("\nWARNING: SCOT matched less than 80% of NB forecast rows.")
        print("Check whether SCOT file has the same forecast window.")

    # -----------------------------
    # 4. Replace baseline SCOT columns
    # -----------------------------
    forecast_df_scot_real["p50_scot"] = forecast_df_scot_real["forecast_qty_p50"]
    forecast_df_scot_real["p70_scot"] = forecast_df_scot_real["forecast_qty_p70"]

    # Keep p70 >= p50 just in case
    forecast_df_scot_real["p70_scot"] = np.maximum(
        forecast_df_scot_real["p70_scot"],
        forecast_df_scot_real["p50_scot"]
    )

    print("\n" + "=" * 80)
    print("FORECAST MEAN CHECK")
    print("=" * 80)

    mean_check = pd.DataFrame([{
        "n_rows": len(forecast_df_scot_real),
        "n_asins": forecast_df_scot_real["asin"].nunique(),
        "true_mean": forecast_df_scot_real["fbi_demand"].mean(),
        "amxl_p50_mean": forecast_df_scot_real["p50_amxl"].mean(),
        "amxl_p70_mean": forecast_df_scot_real["p70_amxl"].mean(),
        "real_scot_p50_mean": forecast_df_scot_real["p50_scot"].mean(),
        "real_scot_p70_mean": forecast_df_scot_real["p70_scot"].mean(),
        "true_zero_rate": (forecast_df_scot_real["fbi_demand"] == 0).mean(),
        "true_active_ratio": (forecast_df_scot_real["fbi_demand"] > 0).mean(),
    }])

    print(mean_check.T)

    # -----------------------------
    # 5. Run WAPE logic
    # -----------------------------
    wape_df = calculate_wape_using_lp_oos2(
        forecast_df_scot_real,
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
    print("FINAL HIGH-SPARSE WAPE WITH REAL SCOT")
    print("=" * 80)

    print("\nP50 WAPE:")
    print(p50_wape)
    print("P50 penalty diff AMXL - SCOT:", p50_penalty_diff)

    print("\nP70 WAPE:")
    print(p70_wape)
    print("P70 penalty diff AMXL - SCOT:", p70_penalty_diff)

    return {
        "forecast_df_scot_real": forecast_df_scot_real,
        "wape_df": wape_df,
        "mean_check": mean_check,
        "p50_wape": p50_wape,
        "p70_wape": p70_wape,
        "p50_penalty_diff": p50_penalty_diff,
        "p70_penalty_diff": p70_penalty_diff,
    }
##
scot_real_outputs = run_high_sparse_scot_alignment_wape(
    result=result,
    scot_df=scot_df,
    remove_oos_dp=True,
    source="lp",
)

forecast_df_scot_real = scot_real_outputs["forecast_df_scot_real"]
wape_df_real_scot = scot_real_outputs["wape_df"]
p50_wape_real_scot = scot_real_outputs["p50_wape"]
p70_wape_real_scot = scot_real_outputs["p70_wape"]
