# ---------------------------------------------
# Healthcare Data Analysis — quick EDA toolkit
# ---------------------------------------------
# Run me from the project root or notebooks/.
# Assumes your CSV lives at: data/hospital_data.csv
#
# Why this structure?
# - Simple, readable prints for a manager or professor.
# - A few charts saved as PNGs in ./reports/figs/.
# - Small helper functions to keep things neat.
#
# Tip: If you're running this in a Jupyter Notebook,
# just paste the cells in order and skip the __main__ part.

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ----------- config -----------
DATA_PATH = Path("C:/Users/ag5488/Documents/VSC/Healthcare_Data_Analysis_Project/data/hospital_data.csv") if Path("notebooks").exists() else Path("data/hospital_data.csv")
FIG_DIR   = Path("reports/figs")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ----------- helpers -----------
def h1(title: str):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))

def h2(title: str):
    print("\n" + title)
    print("-" * len(title))

def fmt_pct(x):
    return f"{x:.1f}%"

def save_bar(ax, filename: str):
    ax.set_xlabel("")
    plt.tight_layout()
    out = FIG_DIR / filename
    plt.savefig(out, dpi=140)
    print(f"[saved] {out.resolve()}")
    plt.close()

# ----------- load -----------
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Oops — I can't find the dataset at: {path.resolve()}")
        print("Make sure hospital_data.csv is in the /data folder.")
        sys.exit(1)
    df = pd.read_csv(path)
    # be explicit with types / parsing for safety:
    for col in ("Admission_Date", "Discharge_Date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# ----------- EDA steps -----------
def basic_overview(df: pd.DataFrame):
    h1("Dataset Overview")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    h2("Columns")
    print(", ".join(df.columns))
    h2("Sample (first 5)")
    print(df.head().to_string(index=False))

def summary_stats(df: pd.DataFrame):
    h1("Summary Statistics")
    numeric_cols = ["Length_of_Stay", "Cost"]
    print(df[numeric_cols].describe().round(2).to_string())

    h2("Missing Values (count)")
    print(df.isna().sum().to_string())

def admissions_by_department(df: pd.DataFrame):
    h1("Admissions by Department")
    counts = df["Department"].value_counts()
    print(counts.to_string())

    # chart
    ax = counts.sort_values(ascending=False).plot(kind="bar", title="Admissions by Department")
    save_bar(ax, "admissions_by_department.png")

def avg_cost_per_department(df: pd.DataFrame):
    h1("Average Cost per Department")
    dept_cost = df.groupby("Department")["Cost"].mean().sort_values(ascending=False).round(2)
    print(dept_cost.to_string())

    ax = dept_cost.plot(kind="bar", title="Average Cost per Patient (by Department)")
    ax.set_ylabel("Average Cost ($)")
    save_bar(ax, "avg_cost_by_department.png")

def readmission_rates(df: pd.DataFrame):
    h1("Readmission Rates")
    overall = (df["Readmission"].value_counts(normalize=True) * 100).round(1)
    print("Overall:")
    print(overall.rename(index=str).apply(lambda v: f"{v:.1f}%").to_string())

    h2("By Department:")
    by_dept = (
        df.groupby("Department")["Readmission"]
          .value_counts(normalize=True)
          .mul(100).rename("pct")
          .reset_index()
          .pivot(index="Department", columns="Readmission", values="pct")
          .fillna(0).round(1)
    )
    # ensure consistent columns if some categories are missing
    for col in ["No", "Yes"]:
        if col not in by_dept.columns:
            by_dept[col] = 0.0
    by_dept = by_dept[["No", "Yes"]]
    print(by_dept.applymap(lambda x: f"{x:.1f}%").to_string())

    # chart (readmission "Yes" by department)
    yes_rates = by_dept["Yes"].sort_values(ascending=False)
    ax = yes_rates.plot(kind="bar", title="Readmission Rate (YES) by Department")
    ax.set_ylabel("Percent of Patients (%)")
    save_bar(ax, "readmission_yes_by_department.png")

def los_vs_cost(df: pd.DataFrame):
    h1("Length of Stay vs Cost")
    corr = df[["Length_of_Stay", "Cost"]].corr().round(3)
    print("Correlation matrix:")
    print(corr.to_string())

    # simple scatter
    ax = df.plot(kind="scatter", x="Length_of_Stay", y="Cost", title="Length of Stay vs. Cost")
    plt.tight_layout()
    out = FIG_DIR / "length_of_stay_vs_cost.png"
    plt.savefig(out, dpi=140)
    print(f"[saved] {out.resolve()}")
    plt.close()

def monthly_admissions(df: pd.DataFrame):
    if "Admission_Date" not in df.columns or df["Admission_Date"].isna().all():
        return
    h1("Monthly Admissions Trend")
    monthly = (
        df.assign(YearMonth=df["Admission_Date"].dt.to_period("M").dt.to_timestamp())
          .groupby("YearMonth")["Patient_ID"]
          .count()
    )
    print(monthly.to_string())

    ax = monthly.plot(kind="line", marker="o", title="Monthly Admissions")
    ax.set_ylabel("Admissions")
    plt.tight_layout()
    out = FIG_DIR / "monthly_admissions.png"
    plt.savefig(out, dpi=140)
    print(f"[saved] {out.resolve()}")
    plt.close()

# ----------- main -----------
def main():
    df = load_data(DATA_PATH)

    # sanity: create a derived LOS if dates exist (optional)
    if "Admission_Date" in df.columns and "Discharge_Date" in df.columns:
        calc_los = (df["Discharge_Date"] - df["Admission_Date"]).dt.days
        # If your CSV already has Length_of_Stay, keep it; otherwise use calc.
        if "Length_of_Stay" not in df.columns or df["Length_of_Stay"].isna().any():
            df["Length_of_Stay"] = calc_los.clip(lower=0)  # avoid negatives if bad data

    basic_overview(df)
    summary_stats(df)
    admissions_by_department(df)
    avg_cost_per_department(df)
    readmission_rates(df)
    los_vs_cost(df)
    monthly_admissions(df)

    h1("Done")
    print("Figures saved under:", FIG_DIR.resolve())

if __name__ == "__main__":
    main()
