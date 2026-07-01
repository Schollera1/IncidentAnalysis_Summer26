# Stats and Graphs File. Looks at incident counts over time: monthly trend, a control chart, a Poisson trend test with a Negative Binomial backup, and a model-free Mann-Kendall test
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from SetUp import ticket_month_year, ticket_priority

priority_filter = "P2 - High"
label_col = "new_l2"
category = None
tickets_df = pd.read_excel("prepped.xlsx")
tickets_df = tickets_df[tickets_df["review_bucket"] != "exclude"]
if priority_filter:
    tickets_df = tickets_df[tickets_df[ticket_priority] == priority_filter]
if category:
    tickets_df = tickets_df[tickets_df[label_col] == category]
tickets_df["month"] = pd.to_datetime(
    tickets_df[ticket_month_year], errors="coerce"
).dt.to_period("M")


# count how many tickets happened each month, filling empty months with 0
def monthly_counts(frame):
    counts = frame.groupby("month").size()
    full_range = pd.period_range(counts.index.min(), counts.index.max(), freq="M")
    counts = counts.reindex(full_range, fill_value=0)
    monthly_counts_df = counts.reset_index()
    monthly_counts_df.columns = ["month", "count"]
    monthly_counts_df["month_index"] = np.arange(len(monthly_counts_df))
    return monthly_counts_df


# monthly count actually going up or down? By how much?
def poisson_trend(monthly_counts_df):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    poisson_model = smf.glm(
        "count ~ month_index", data=monthly_counts_df, family=sm.families.Poisson()
    ).fit()
    trend_slope = poisson_model.params["month_index"]
    p_value = poisson_model.pvalues["month_index"]
    print("=== Poisson trend ===")
    print(
        f"rate change per month : {100 * (np.exp(trend_slope) - 1):+.1f}%   (p={p_value:.4f})"
    )
    dispersion = poisson_model.pearson_chi2 / poisson_model.df_resid
    print(
        f"dispersion            : {dispersion:.2f}  (want about 1, above 1.5 means use NegBin)"
    )
    if dispersion > 1.5:
        negbin_model = smf.glm(
            "count ~ month_index",
            data=monthly_counts_df,
            family=sm.families.NegativeBinomial(),
        ).fit()
        trend_slope = negbin_model.params["month_index"]
        p_value = negbin_model.pvalues["month_index"]
        print(
            f"[NegBin] rate change per month: {100 * (np.exp(trend_slope) - 1):+.1f}%  (p={p_value:.4f})"
        )


# Numerical analysis for above
def mann_kendall(monthly_counts_df):
    try:
        import pymannkendall as mk
    except ImportError:
        print("\n(install pymannkendall to get the model-free trend test)")
        return
    result = mk.original_test(monthly_counts_df["count"].values)
    print("\n=== Mann-Kendall (model-free) ===")
    print(
        f"trend = {result.trend}   p = {result.p:.4f}   slope per month = {result.slope:.2f}"
    )


# Plots counts across time plus/minus 3 sigmas with central mean
def control_chart(monthly_counts_df, output_file="monthly_control_chart.png"):
    values = monthly_counts_df["count"].values
    mean_count = values.mean()
    upper = mean_count + 3 * np.sqrt(mean_count)
    lower = max(0, mean_count - 3 * np.sqrt(mean_count))
    months = monthly_counts_df["month"].astype(str)
    point_colors = [
        "#C4442E" if (v > upper or v < lower) else "#0070E0" for v in values
    ]  # red = out of limits
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(months, values, color="#0070E0", linewidth=1, zorder=1)
    axis.scatter(months, values, c=point_colors, zorder=2)
    axis.axhline(mean_count, linestyle="--", color="#001435")
    axis.axhline(upper, linestyle=":", color="#C4442E")
    axis.axhline(lower, linestyle=":", color="#C4442E")
    axis.set_title("Monthly incidents, control chart")
    axis.tick_params(axis="x", rotation=90)
    figure.tight_layout()
    figure.savefig(output_file, dpi=120)
    plt.close()
    print(f"\nSaved {output_file}")


# Bar chart of top categories by volume
def top_categories(output_file="top_categories.png"):
    counts = tickets_df[label_col].value_counts().head(10)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.barh(counts.index[::-1].astype(str), counts.values[::-1], color="#0070E0")
    axis.set_title(f"Top categories by volume ({label_col})")
    figure.tight_layout()
    figure.savefig(output_file, dpi=120)
    plt.close()
    print(f"Saved {output_file}")


if __name__ == "__main__":
    monthly_counts_df = monthly_counts(tickets_df)
    print(monthly_counts_df.to_string(index=False))
    poisson_trend(monthly_counts_df)
    mann_kendall(monthly_counts_df)
    control_chart(monthly_counts_df)
    if not category:
        top_categories()
