forecast_df = result["forecast_df"].copy()
forecast_df["asin"] = forecast_df["asin"].astype(str).str.strip()

scot_check = scot_df.copy()
scot_check["asin"] = scot_check["asin"].astype(str).str.strip()

nb_asins = set(forecast_df["asin"].unique())
scot_asins = set(scot_check["asin"].unique())

print("NB high-sparse ASINs:", len(nb_asins))
print("SCOT ASINs:", len(scot_asins))
print("Intersection ASINs:", len(nb_asins & scot_asins))
print("NB missing in SCOT:", len(nb_asins - scot_asins))

print("Example missing ASINs:")
print(list(nb_asins - scot_asins)[:20])
