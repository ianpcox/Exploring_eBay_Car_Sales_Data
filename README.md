# Market Intelligence: German Used Car Market (Project Elevate)

This repository transforms a basic exploratory data analysis (EDA) notebook into a **sophisticated market intelligence platform** analyzing 50,000 used car listings from eBay Kleinanzeigen.

By moving beyond simple averages, this project implements robust price anomaly detection, brand-level market segmentation, and multivariate price driver analysis to uncover the true mechanics of the German secondary auto market.

## Project Structure

* `ebay_pipeline.py` — The core reproducible data pipeline. Performs deep cleaning, IQR anomaly detection, and generates 12 static charts plus an interactive dashboard.
* `docs/report.md` — A comprehensive paper-style report detailing the methodology, market segmentation, and drivers of depreciation.
* `docs/dashboard.html` — An **interactive Plotly dashboard** exploring the market data.
* `docs/assets/` — 12 generated static charts supporting the report.
* `Exploring eBay Car Sales Data.ipynb` — The original legacy EDA notebook.

## Key Findings

1. **Brand Origin Matters:** Domestic brands (VW, BMW, Audi, Mercedes) completely dominate both the volume and premium segments of the German market.
2. **The 100k Cliff:** Vehicles lose the vast majority of their premium pricing power once they cross 100,000 km or 7 years of age.
3. **Condition is King:** Unrepaired damage destroys ~73% of a vehicle's residual value on the secondary market.
4. **Transmission Premium:** Across almost all top brands, automatic vehicles command a significantly higher median price than manual counterparts.

Read the full analysis in [docs/report.md](docs/report.md) or open `docs/dashboard.html` in your browser to explore the data interactively.

## How to Run the Pipeline

Install the dependencies:
```bash
pip install pandas numpy matplotlib seaborn scipy plotly
```

Run the pipeline:
```bash
python ebay_pipeline.py
```
This will process the data, detect anomalies, and regenerate all charts and the dashboard in the `ebay_outputs/` directory.
