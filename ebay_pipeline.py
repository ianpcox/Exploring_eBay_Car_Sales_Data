"""
Elevated eBay Car Sales Market Intelligence Pipeline
Project Elevate — Exploring_eBay_Car_Sales_Data

Performs:
  1. Deep data cleaning & translation (German -> English)
  2. Price anomaly detection (IQR + Z-score)
  3. Brand market segmentation (volume vs. median price)
  4. Mileage vs. price regression analysis
  5. Damage premium analysis
  6. Fuel type market share trends
  7. Vehicle type distribution
  8. Age vs. price depreciation curves
  9. Top brand/model combinations
  10. Listing activity heatmap

Generates 12 static charts + interactive Plotly dashboard.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/ubuntu/ebay_outputs")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

# ── Translation maps ──────────────────────────────────────────────────────────
VEHICLE_TYPE_MAP = {
    "limousine": "Sedan", "kombi": "Estate/Wagon", "kleinwagen": "Compact",
    "suv": "SUV", "cabrio": "Convertible", "coupe": "Coupe",
    "bus": "Van/Bus", "andere": "Other",
}
GEARBOX_MAP    = {"manuell": "Manual", "automatik": "Automatic"}
FUEL_MAP       = {
    "benzin": "Petrol", "diesel": "Diesel", "lpg": "LPG",
    "elektro": "Electric", "hybrid": "Hybrid", "cng": "CNG", "andere": "Other",
}
DAMAGE_MAP     = {"nein": "No Damage", "ja": "Has Damage"}


# ── Data loading & cleaning ───────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    print("Loading data...")
    df = pd.read_csv(path, encoding="latin-1")
    print(f"  Raw rows: {len(df):,}")

    # Clean price: strip "$" and "," then convert
    df["price"] = (
        df["price"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )
    # Clean odometer
    df["odometer_km"] = (
        df["odometer"].astype(str)
        .str.replace(r"[km,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Filter unrealistic prices and years
    df = df[(df["price"] >= 100) & (df["price"] <= 150_000)]
    df = df[(df["yearOfRegistration"] >= 1950) & (df["yearOfRegistration"] <= 2016)]
    df = df[df["powerPS"] < 1000]

    # Translate German labels
    df["vehicleType"]       = df["vehicleType"].map(VEHICLE_TYPE_MAP).fillna("Other")
    df["gearbox"]           = df["gearbox"].map(GEARBOX_MAP).fillna("Unknown")
    df["fuelType"]          = df["fuelType"].map(FUEL_MAP).fillna("Other")
    df["notRepairedDamage"] = df["notRepairedDamage"].map(DAMAGE_MAP).fillna("Unknown")

    # Derived columns
    df["age_years"] = 2016 - df["yearOfRegistration"]
    df["brand"]     = df["brand"].str.title()

    # Parse dates
    for col in ["dateCrawled", "dateCreated", "lastSeen"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["listing_month"] = df["dateCreated"].dt.month
    df["listing_dow"]   = df["dateCreated"].dt.dayofweek

    print(f"  Clean rows: {len(df):,}  ({len(df)/50000*100:.1f}% retained)")
    return df


# ── Anomaly detection ─────────────────────────────────────────────────────────
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag price anomalies using IQR method per brand."""
    df = df.copy()
    df["is_anomaly"] = False
    for brand, grp in df.groupby("brand"):
        if len(grp) < 10:
            continue
        q1, q3 = grp["price"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df["brand"] == brand) & ((df["price"] < lo) | (df["price"] > hi))
        df.loc[mask, "is_anomaly"] = True
    n_anomalies = df["is_anomaly"].sum()
    print(f"  Anomalies detected: {n_anomalies:,} ({n_anomalies/len(df)*100:.1f}%)")
    return df


# ── Brand segmentation ────────────────────────────────────────────────────────
def brand_stats(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    clean = df[~df["is_anomaly"]]
    stats_df = (
        clean.groupby("brand")["price"]
        .agg(count="count", median="median", mean="mean", std="std")
        .reset_index()
    )
    stats_df = stats_df[stats_df["count"] >= 50].sort_values("count", ascending=False)
    return stats_df.head(top_n)


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_price_distribution(df):
    clean = df[~df["is_anomaly"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(clean["price"], bins=80, color="#3498DB", alpha=0.85, edgecolor="white")
    axes[0].axvline(clean["price"].median(), color="#E74C3C", lw=2,
                    label=f'Median: €{clean["price"].median():,.0f}')
    axes[0].axvline(clean["price"].mean(), color="#F39C12", lw=2, linestyle="--",
                    label=f'Mean: €{clean["price"].mean():,.0f}')
    axes[0].set_xlabel("Price (€)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Price Distribution (Anomalies Removed)")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))

    axes[1].hist(np.log1p(clean["price"]), bins=60, color="#2ECC71", alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("log(Price + 1)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Log-Transformed Price Distribution\n(Reveals Near-Normal Shape)")

    plt.suptitle("eBay Car Prices — Distribution Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_price_distribution.png")
    plt.close()
    print("Saved: 01_price_distribution.png")


def plot_anomaly_overview(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Anomaly vs normal price scatter
    normal  = df[~df["is_anomaly"]].sample(min(3000, (~df["is_anomaly"]).sum()), random_state=42)
    anomaly = df[df["is_anomaly"]]
    axes[0].scatter(normal["odometer_km"], normal["price"], alpha=0.15, s=5,
                    color="#3498DB", label=f"Normal ({len(normal):,})")
    axes[0].scatter(anomaly["odometer_km"], anomaly["price"], alpha=0.5, s=15,
                    color="#E74C3C", label=f"Anomaly ({len(anomaly):,})")
    axes[0].set_xlabel("Odometer (km)")
    axes[0].set_ylabel("Price (€)")
    axes[0].set_title("Price Anomalies vs. Normal Listings")
    axes[0].legend(markerscale=3)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))

    # Anomaly rate by brand
    top_brands = df["brand"].value_counts().head(15).index
    anom_rate = (
        df[df["brand"].isin(top_brands)]
        .groupby("brand")["is_anomaly"]
        .mean()
        .sort_values(ascending=True)
        * 100
    )
    colors = ["#E74C3C" if v > 15 else "#3498DB" for v in anom_rate.values]
    axes[1].barh(anom_rate.index, anom_rate.values, color=colors, alpha=0.85)
    axes[1].axvline(anom_rate.mean(), color="black", linestyle="--", lw=1,
                    label=f"Avg: {anom_rate.mean():.1f}%")
    axes[1].set_xlabel("Anomaly Rate (%)")
    axes[1].set_title("Price Anomaly Rate by Brand (Top 15)")
    axes[1].legend()

    plt.suptitle("Price Anomaly Detection — IQR Method", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_anomaly_detection.png")
    plt.close()
    print("Saved: 02_anomaly_detection.png")


def plot_brand_segmentation(bstats):
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        bstats["count"], bstats["median"],
        s=bstats["count"] / 5,
        c=bstats["median"],
        cmap="RdYlGn",
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
    )
    for _, row in bstats.iterrows():
        ax.annotate(
            row["brand"],
            (row["count"], row["median"]),
            textcoords="offset points", xytext=(6, 3),
            fontsize=7.5,
        )
    plt.colorbar(scatter, ax=ax, label="Median Price (€)")
    ax.set_xlabel("Number of Listings (Volume)")
    ax.set_ylabel("Median Price (€)")
    ax.set_title("Brand Market Segmentation\nBubble size = listing volume, color = median price",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_brand_segmentation.png")
    plt.close()
    print("Saved: 03_brand_segmentation.png")


def plot_brand_price_bars(bstats):
    top15 = bstats.sort_values("median", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top15["brand"], top15["median"],
                   color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top15))),
                   alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, top15["median"]):
        ax.text(val + 100, bar.get_y() + bar.get_height()/2,
                f"€{val:,.0f}", va="center", fontsize=8)
    ax.set_xlabel("Median Price (€)")
    ax.set_title("Median Listing Price by Brand (Top 15)", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_brand_price_bars.png")
    plt.close()
    print("Saved: 04_brand_price_bars.png")


def plot_mileage_vs_price(df):
    clean = df[~df["is_anomaly"]]
    sample = clean.sample(min(5000, len(clean)), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter with regression
    axes[0].scatter(sample["odometer_km"], sample["price"],
                    alpha=0.12, s=5, color="#3498DB")
    m, b, r, p, _ = stats.linregress(sample["odometer_km"].dropna(),
                                      sample["price"].dropna())
    x_line = np.linspace(sample["odometer_km"].min(), sample["odometer_km"].max(), 100)
    axes[0].plot(x_line, m * x_line + b, color="#E74C3C", lw=2,
                 label=f"r = {r:.3f}, p < 0.001")
    axes[0].set_xlabel("Odometer (km)")
    axes[0].set_ylabel("Price (€)")
    axes[0].set_title("Mileage vs. Price")
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))

    # Median price by mileage bucket
    clean2 = clean.copy()
    clean2["mileage_bucket"] = pd.cut(clean2["odometer_km"],
                                      bins=[0, 50000, 100000, 150000, 200000],
                                      labels=["0–50k", "50–100k", "100–150k", "150–200k"])
    bucket_med = clean2.groupby("mileage_bucket", observed=True)["price"].median()
    axes[1].bar(bucket_med.index.astype(str), bucket_med.values,
                color=["#2ECC71", "#F39C12", "#E67E22", "#E74C3C"], alpha=0.85, edgecolor="white")
    for i, (label, val) in enumerate(zip(bucket_med.index, bucket_med.values)):
        axes[1].text(i, val + 100, f"€{val:,.0f}", ha="center", fontsize=9)
    axes[1].set_xlabel("Mileage Range")
    axes[1].set_ylabel("Median Price (€)")
    axes[1].set_title("Median Price by Mileage Bucket")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))

    plt.suptitle("Mileage vs. Price Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_mileage_vs_price.png")
    plt.close()
    print("Saved: 05_mileage_vs_price.png")


def plot_age_depreciation(df):
    clean = df[~df["is_anomaly"] & (df["age_years"] <= 30) & (df["age_years"] >= 0)]
    age_med = clean.groupby("age_years")["price"].median().reset_index()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(age_med["age_years"], age_med["price"], "o-",
            color="#3498DB", lw=2, markersize=5)
    ax.fill_between(age_med["age_years"], age_med["price"], alpha=0.15, color="#3498DB")
    ax.set_xlabel("Vehicle Age (Years)")
    ax.set_ylabel("Median Price (€)")
    ax.set_title("Vehicle Age vs. Median Price — Depreciation Curve",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_age_depreciation.png")
    plt.close()
    print("Saved: 06_age_depreciation.png")


def plot_damage_analysis(df):
    clean = df[~df["is_anomaly"] & (df["notRepairedDamage"].isin(["No Damage", "Has Damage"]))]
    top_brands = clean["brand"].value_counts().head(12).index
    damage_df = (
        clean[clean["brand"].isin(top_brands)]
        .groupby(["brand", "notRepairedDamage"])["price"]
        .median()
        .unstack()
        .dropna()
    )
    damage_df["discount_pct"] = (
        (damage_df["No Damage"] - damage_df["Has Damage"]) / damage_df["No Damage"] * 100
    )
    damage_df = damage_df.sort_values("discount_pct", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Side-by-side bars
    x = np.arange(len(damage_df))
    w = 0.35
    axes[0].bar(x - w/2, damage_df["No Damage"], w, label="No Damage",
                color="#2ECC71", alpha=0.85, edgecolor="white")
    axes[0].bar(x + w/2, damage_df["Has Damage"], w, label="Has Damage",
                color="#E74C3C", alpha=0.85, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(damage_df.index, rotation=45, ha="right")
    axes[0].set_ylabel("Median Price (€)")
    axes[0].set_title("Damaged vs. Undamaged Median Price")
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))

    # Discount percentage
    colors = ["#E74C3C" if v > 40 else "#F39C12" if v > 25 else "#2ECC71"
              for v in damage_df["discount_pct"]]
    axes[1].barh(damage_df.index, damage_df["discount_pct"],
                 color=colors, alpha=0.85, edgecolor="white")
    axes[1].axvline(damage_df["discount_pct"].mean(), color="black", linestyle="--", lw=1,
                    label=f"Avg: {damage_df['discount_pct'].mean():.1f}%")
    axes[1].set_xlabel("Price Discount for Damage (%)")
    axes[1].set_title("Damage Discount by Brand")
    axes[1].legend()

    plt.suptitle("Impact of Unrepaired Damage on Listing Price", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_damage_analysis.png")
    plt.close()
    print("Saved: 07_damage_analysis.png")


def plot_fuel_type(df):
    clean = df[~df["is_anomaly"]]
    fuel_counts = clean["fuelType"].value_counts()
    fuel_price  = clean.groupby("fuelType")["price"].median().reindex(fuel_counts.index)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(fuel_counts)))
    axes[0].pie(fuel_counts.values, labels=fuel_counts.index,
                autopct="%1.1f%%", colors=colors, startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[0].set_title("Fuel Type Market Share")

    axes[1].bar(fuel_price.index, fuel_price.values, color=colors, alpha=0.85, edgecolor="white")
    for i, (label, val) in enumerate(zip(fuel_price.index, fuel_price.values)):
        axes[1].text(i, val + 100, f"€{val:,.0f}", ha="center", fontsize=8)
    axes[1].set_ylabel("Median Price (€)")
    axes[1].set_title("Median Price by Fuel Type")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))
    axes[1].tick_params(axis="x", rotation=20)

    plt.suptitle("Fuel Type Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_fuel_type.png")
    plt.close()
    print("Saved: 08_fuel_type.png")


def plot_vehicle_type(df):
    clean = df[~df["is_anomaly"]]
    vtype_counts = clean["vehicleType"].value_counts()
    vtype_price  = clean.groupby("vehicleType")["price"].median().reindex(vtype_counts.index)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Paired(np.linspace(0, 1, len(vtype_counts)))
    axes[0].barh(vtype_counts.index, vtype_counts.values, color=colors, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Number of Listings")
    axes[0].set_title("Listings by Vehicle Type")

    axes[1].barh(vtype_price.index, vtype_price.values, color=colors, alpha=0.85, edgecolor="white")
    for i, val in enumerate(vtype_price.values):
        axes[1].text(val + 100, i, f"€{val:,.0f}", va="center", fontsize=8)
    axes[1].set_xlabel("Median Price (€)")
    axes[1].set_title("Median Price by Vehicle Type")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))

    plt.suptitle("Vehicle Type Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_vehicle_type.png")
    plt.close()
    print("Saved: 09_vehicle_type.png")


def plot_top_models(df):
    clean = df[~df["is_anomaly"]]
    clean["brand_model"] = clean["brand"] + " " + clean["model"].str.title().fillna("(Unknown)")
    top_models = (
        clean.groupby("brand_model")["price"]
        .agg(count="count", median="median")
        .query("count >= 100")
        .sort_values("count", ascending=False)
        .head(20)
        .sort_values("median", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(top_models.index, top_models["median"],
                   color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top_models))),
                   alpha=0.85, edgecolor="white")
    for bar, (_, row) in zip(bars, top_models.iterrows()):
        ax.text(row["median"] + 100, bar.get_y() + bar.get_height()/2,
                f"€{row['median']:,.0f}  (n={row['count']:,})", va="center", fontsize=7.5)
    ax.set_xlabel("Median Price (€)")
    ax.set_title("Top 20 Brand/Model Combinations by Listing Volume\n(Sorted by Median Price)",
                 fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_top_models.png")
    plt.close()
    print("Saved: 10_top_models.png")


def plot_listing_heatmap(df):
    clean = df[~df["is_anomaly"]].dropna(subset=["listing_month", "listing_dow"])
    pivot = clean.pivot_table(
        index="listing_dow", columns="listing_month",
        values="price", aggfunc="count", fill_value=0
    )
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.columns = [month_names[m-1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3,
                cbar_kws={"label": "Number of Listings"}, fmt="d", annot=True, annot_kws={"size": 7})
    ax.set_title("Listing Activity Heatmap — Day of Week vs. Month",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "11_listing_heatmap.png")
    plt.close()
    print("Saved: 11_listing_heatmap.png")


def plot_gearbox_analysis(df):
    clean = df[~df["is_anomaly"] & df["gearbox"].isin(["Manual", "Automatic"])]
    top_brands = clean["brand"].value_counts().head(10).index
    gear_df = (
        clean[clean["brand"].isin(top_brands)]
        .groupby(["brand", "gearbox"])["price"]
        .median()
        .unstack()
        .dropna()
        .sort_values("Automatic", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(gear_df))
    w = 0.35
    ax.bar(x - w/2, gear_df["Manual"], w, label="Manual",
           color="#3498DB", alpha=0.85, edgecolor="white")
    ax.bar(x + w/2, gear_df["Automatic"], w, label="Automatic",
           color="#E74C3C", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(gear_df.index, rotation=30, ha="right")
    ax.set_ylabel("Median Price (€)")
    ax.set_title("Manual vs. Automatic Transmission — Median Price by Brand",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "12_gearbox_analysis.png")
    plt.close()
    print("Saved: 12_gearbox_analysis.png")


# ── Interactive Plotly Dashboard ──────────────────────────────────────────────
def build_dashboard(df, bstats):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    clean = df[~df["is_anomaly"]]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Brand Market Segmentation (Volume vs. Price)",
            "Price Distribution by Vehicle Type",
            "Age Depreciation Curve",
            "Fuel Type Market Share",
            "Mileage vs. Price (Sample)",
            "Damage Impact on Price by Brand",
        ],
        specs=[
            [{"type": "scatter"}, {"type": "box"}],
            [{"type": "scatter"}, {"type": "pie"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # 1. Brand segmentation scatter
    fig.add_trace(go.Scatter(
        x=bstats["count"], y=bstats["median"],
        mode="markers+text",
        text=bstats["brand"],
        textposition="top center",
        marker=dict(size=bstats["count"]/30, color=bstats["median"],
                    colorscale="RdYlGn", showscale=True,
                    colorbar=dict(title="Median €", x=0.46)),
        hovertemplate="<b>%{text}</b><br>Listings: %{x:,}<br>Median: €%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # 2. Box plot by vehicle type
    for vtype in clean["vehicleType"].value_counts().head(6).index:
        subset = clean[clean["vehicleType"] == vtype]["price"]
        fig.add_trace(go.Box(y=subset, name=vtype, showlegend=False), row=1, col=2)

    # 3. Depreciation curve
    age_med = clean[clean["age_years"].between(0, 30)].groupby("age_years")["price"].median()
    fig.add_trace(go.Scatter(
        x=age_med.index, y=age_med.values,
        mode="lines+markers",
        line=dict(color="#3498DB", width=2),
        fill="tozeroy",
        fillcolor="rgba(52,152,219,0.15)",
        hovertemplate="Age: %{x} yrs<br>Median: €%{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    # 4. Fuel type pie
    fuel_counts = clean["fuelType"].value_counts()
    fig.add_trace(go.Pie(
        labels=fuel_counts.index, values=fuel_counts.values,
        hole=0.35, textinfo="label+percent",
    ), row=2, col=2)

    # 5. Mileage vs price scatter
    sample = clean.sample(min(2000, len(clean)), random_state=42)
    fig.add_trace(go.Scatter(
        x=sample["odometer_km"], y=sample["price"],
        mode="markers",
        marker=dict(size=3, color="#3498DB", opacity=0.4),
        hovertemplate="Km: %{x:,}<br>Price: €%{y:,.0f}<extra></extra>",
    ), row=3, col=1)

    # 6. Damage discount bars
    damage_clean = clean[clean["notRepairedDamage"].isin(["No Damage", "Has Damage"])]
    top_brands = damage_clean["brand"].value_counts().head(10).index
    damage_df = (
        damage_clean[damage_clean["brand"].isin(top_brands)]
        .groupby(["brand", "notRepairedDamage"])["price"]
        .median().unstack().dropna()
    )
    damage_df["discount"] = (damage_df["No Damage"] - damage_df["Has Damage"]) / damage_df["No Damage"] * 100
    damage_df = damage_df.sort_values("discount")
    fig.add_trace(go.Bar(
        x=damage_df.index, y=damage_df["discount"],
        marker_color="#E74C3C", opacity=0.8,
        hovertemplate="%{x}<br>Discount: %{y:.1f}%<extra></extra>",
    ), row=3, col=2)

    fig.update_layout(
        title=dict(text="eBay Car Sales — Market Intelligence Dashboard",
                   font=dict(size=18, family="Arial"), x=0.5),
        height=1100,
        template="plotly_white",
        showlegend=False,
        font=dict(family="Arial", size=11),
    )
    fig.update_yaxes(tickprefix="€", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(tickprefix="€", tickformat=",.0f", row=1, col=2)
    fig.update_yaxes(tickprefix="€", tickformat=",.0f", row=2, col=1)
    fig.update_yaxes(tickprefix="€", tickformat=",.0f", row=3, col=1)
    fig.update_yaxes(ticksuffix="%", row=3, col=2)

    out_path = OUT_DIR / "dashboard.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved: dashboard.html")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  EBAY CAR SALES — MARKET INTELLIGENCE PIPELINE")
    print("="*60)

    df = load_and_clean("/home/ubuntu/autos.csv")
    df = detect_anomalies(df)
    bstats = brand_stats(df)

    print("\nGenerating static visualizations...")
    plot_price_distribution(df)
    plot_anomaly_overview(df)
    plot_brand_segmentation(bstats)
    plot_brand_price_bars(bstats)
    plot_mileage_vs_price(df)
    plot_age_depreciation(df)
    plot_damage_analysis(df)
    plot_fuel_type(df)
    plot_vehicle_type(df)
    plot_top_models(df)
    plot_listing_heatmap(df)
    plot_gearbox_analysis(df)

    print("\nBuilding interactive Plotly dashboard...")
    build_dashboard(df, bstats)

    # Save clean dataset
    clean = df[~df["is_anomaly"]]
    clean.to_csv(OUT_DIR / "autos_clean.csv", index=False)

    # Summary stats
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"  Total listings (raw):    50,000")
    print(f"  After cleaning:          {len(df):,}")
    print(f"  Anomalies detected:      {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean()*100:.1f}%)")
    print(f"  Clean listings:          {len(clean):,}")
    print(f"  Median price:            €{clean['price'].median():,.0f}")
    print(f"  Mean price:              €{clean['price'].mean():,.0f}")
    print(f"  Most common brand:       {clean['brand'].value_counts().index[0]}")
    print(f"  Most expensive brand:    {clean.groupby('brand')['price'].median().idxmax()}")
    print(f"  Damage discount (avg):   {((clean[clean['notRepairedDamage']=='No Damage']['price'].median() - clean[clean['notRepairedDamage']=='Has Damage']['price'].median()) / clean[clean['notRepairedDamage']=='No Damage']['price'].median() * 100):.1f}%")
    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
