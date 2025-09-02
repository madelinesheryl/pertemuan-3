import pandas as pd
import numpy as np

# ==== CONFIG ====
INPUT_PATH = "TGM 2020-2023_eng.csv.xlsx"
SHEET_NAME = "TGM 2020-2023_eng"
IQR_K = 1.5                       # Tukey rule multiplier
EXCLUDE_COLS = {"Year"}           # columns to skip for outlier detection
OUTPUT_XLSX = "TGM_2020_2023_cleaned_IQR.xlsx"
OUTPUT_CSV  = "TGM_2020_2023_cleaned_IQR.csv"

# ==== LOAD ====
df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)

# ==== MISSING VALUES SUMMARY ====
missing_summary = df.isna().sum().rename("missing_count").to_frame()
missing_summary["pct"] = (missing_summary["missing_count"] / len(df)).round(4)
print("=== Missing values per column ===")
print(missing_summary, "\n")

# ==== NUMERIC COLUMNS ====
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]
print("Numeric columns used for IQR:", numeric_cols, "\n")

# ==== IQR HELPERS ====
def iqr_bounds(series, k=1.5):
    """Return (lower, upper) bounds using IQR. Ignores NaNs.
       If IQR == 0 (constant/near-constant), return (None, None) to skip."""
    s = series.dropna()
    if s.empty:
        return (None, None)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return (None, None)
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (lower, upper)

# ==== OUTLIER FLAGS PER COLUMN ====
outlier_flags = pd.DataFrame(index=df.index)
bounds_info = []
for col in numeric_cols:
    lo, up = iqr_bounds(df[col], k=IQR_K)
    if lo is None or up is None:
        # skip constant / all-NaN columns
        outlier_flags[col] = False
        bounds_info.append({"column": col, "lower": None, "upper": None, "iqr_k": IQR_K, "skipped": True})
        continue
    flags = (df[col] < lo) | (df[col] > up)
    flags = flags.fillna(False)  # NaN is NOT an outlier
    outlier_flags[col] = flags
    bounds_info.append({"column": col, "lower": lo, "upper": up, "iqr_k": IQR_K, "skipped": False})

bounds_df = pd.DataFrame(bounds_info)
outlier_counts = outlier_flags.sum().rename("outlier_count").to_frame()
outlier_counts["pct"] = (outlier_counts["outlier_count"] / len(df)).round(4)

print("=== Outlier counts per numeric column (IQR) ===")
print(outlier_counts, "\n")

# ==== DROP ROWS THAT HAVE ANY OUTLIER (across selected numeric cols) ====
any_outlier = outlier_flags.any(axis=1)
n_before = df.shape[0]
cleaned = df.loc[~any_outlier].copy()
n_after = cleaned.shape[0]

print(f"Rows before: {n_before}")
print(f"Rows after : {n_after}")
print(f"Dropped    : {n_before - n_after}\n")

# ==== SAVE OUTPUTS ====
with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name="original")
    missing_summary.to_excel(writer, sheet_name="missing_summary")
    bounds_df.to_excel(writer, index=False, sheet_name="iqr_bounds")
    outlier_counts.to_excel(writer, sheet_name="outlier_counts")
    outlier_flags.to_excel(writer, sheet_name="outlier_flags")
    cleaned.to_excel(writer, index=False, sheet_name="cleaned")

cleaned.to_csv(OUTPUT_CSV, index=False)
print(f"Saved Excel to: {OUTPUT_XLSX}")
print(f"Saved CSV   to: {OUTPUT_CSV}")
