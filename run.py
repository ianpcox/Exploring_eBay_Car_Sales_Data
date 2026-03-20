"""
Entry point for Exploring_eBay_Car_Sales_Data. Baseline: median price by make.
"""
import os
import pandas as pd

def main():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "autos.csv")
    if not os.path.isfile(path):
        print("=== Exploring_eBay_Car_Sales_Data ===\nRun from project root. Run notebook for full analysis.")
        return
    df = pd.read_csv(path)
    print("=== Exploring_eBay_Car_Sales_Data ===\nRows:", len(df))
    price_col = "price" if "price" in df.columns else df.columns[df.dtypes == "object"].tolist()[0] if len(df.columns) else None
    if price_col and pd.api.types.is_numeric_dtype(df[price_col]):
        baseline = df[price_col].median()
        print("Baseline (median price):", baseline)
    print("Full analysis: run 'Exploring eBay Car Sales Data.ipynb'.")

if __name__ == "__main__":
    main()
