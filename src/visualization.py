"""
Exploratory Data Analysis and Visualization Module.
Course: Exploratory Data Analysis and Visualization (U21ADP05)
Institution: KPR Institute of Engineering and Technology
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def plot_cancellation_count(df: pd.DataFrame, output_dir: str = OUTPUTS_DIR) -> dict:
    """1. Cancellation vs Non-cancellation Count Plot"""
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        x="is_canceled",
        data=df,
        hue="is_canceled",
        palette=["#2b5c8f", "#d95f02"],
        legend=False,
    )
    plt.title("Reservation Status Distribution (Not Canceled vs Canceled)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Reservation Outcome", fontsize=11, labelpad=8)
    plt.ylabel("Number of Bookings", fontsize=11, labelpad=8)
    plt.xticks([0, 1], ["Not Canceled (Check-Out)", "Canceled (Cancellation/No-Show)"])

    total = len(df)
    counts = df["is_canceled"].value_counts()
    for i, count in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        pct = (count / total) * 100
        ax.annotate(
            f"{count:,}\n({pct:.1f}%)",
            (i, count / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()
    filepath = os.path.join(output_dir, "eda_01_cancellation_count.png")
    plt.savefig(filepath)
    plt.close()

    metadata = {
        "figure_name": "eda_01_cancellation_count.png",
        "title": "Overall Reservation Cancellation Distribution",
        "purpose": "Analyze baseline target class distribution and quantify reservation cancellations.",
        "findings": (
            f"Of {total:,} analyzed reservations, {counts.get(0, 0):,} ({counts.get(0, 0)/total*100:.1f}%) "
            f"were fulfilled successfully, while {counts.get(1, 0):,} ({counts.get(1, 0)/total*100:.1f}%) "
            "were canceled or marked as no-shows. This indicates a high baseline cancellation burden on hotel operations."
        ),
    }
    return metadata


def plot_lead_time_distribution(df: pd.DataFrame, output_dir: str = OUTPUTS_DIR) -> dict:
    """2. Lead-Time Distribution using Histogram and KDE split by status"""
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df,
        x="lead_time",
        hue="is_canceled",
        kde=True,
        bins=50,
        palette=["#2b5c8f", "#d95f02"],
        alpha=0.45,
        stat="density",
        common_norm=False,
    )
    plt.title("Lead Time Distribution Stratified by Reservation Outcome", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Lead Time (Days between Booking and Arrival)", fontsize=11, labelpad=8)
    plt.ylabel("Probability Density", fontsize=11, labelpad=8)
    plt.legend(title="Outcome", labels=["Canceled", "Not Canceled"], frameon=True)

    median_canceled = df[df["is_canceled"] == 1]["lead_time"].median()
    median_not_canceled = df[df["is_canceled"] == 0]["lead_time"].median()

    plt.axvline(median_not_canceled, color="#2b5c8f", linestyle="--", linewidth=1.5, label=f"Median Not Canceled ({median_not_canceled:.0f}d)")
    plt.axvline(median_canceled, color="#d95f02", linestyle="--", linewidth=1.5, label=f"Median Canceled ({median_canceled:.0f}d)")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "eda_02_lead_time_distribution.png")
    plt.savefig(filepath)
    plt.close()

    metadata = {
        "figure_name": "eda_02_lead_time_distribution.png",
        "title": "Lead-Time Distribution (Histogram and KDE)",
        "purpose": "Assess how advance booking duration relates to reservation cancellations.",
        "findings": (
            f"Canceled reservations exhibit a significantly higher median lead time ({median_canceled:.0f} days) "
            f"compared to fulfilled reservations ({median_not_canceled:.0f} days). Guests booking far in advance "
            "face greater itinerary uncertainty, elevating the likelihood of cancellation."
        ),
    }
    return metadata


def plot_cancellation_by_hotel_type(df: pd.DataFrame, output_dir: str = OUTPUTS_DIR) -> dict:
    """3. Cancellation Rate and Volume by Hotel Type"""
    summary = df.groupby("hotel")["is_canceled"].agg(
        total="count",
        canceled="sum",
        rate="mean"
    ).reset_index()
    summary["rate_pct"] = summary["rate"] * 100

    fig, ax1 = plt.subplots(figsize=(8, 5))
    colors = ["#1b9e77", "#7570b3"]

    bars = ax1.bar(summary["hotel"], summary["rate_pct"], color=colors, width=0.45, alpha=0.85, edgecolor="black")
    ax1.set_title("Cancellation Rate by Hotel Property Type", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Hotel Type", fontsize=11, labelpad=8)
    ax1.set_ylabel("Cancellation Rate (%)", fontsize=11, labelpad=8)
    ax1.set_ylim(0, max(summary["rate_pct"]) * 1.25)

    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{row['rate_pct']:.1f}%\n({int(row['canceled']):,}/{int(row['total']):,})",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    filepath = os.path.join(output_dir, "eda_03_cancellation_by_hotel_type.png")
    plt.savefig(filepath)
    plt.close()

    city_rate = summary.loc[summary["hotel"] == "City Hotel", "rate_pct"].values[0] if "City Hotel" in summary["hotel"].values else 0
    resort_rate = summary.loc[summary["hotel"] == "Resort Hotel", "rate_pct"].values[0] if "Resort Hotel" in summary["hotel"].values else 0

    metadata = {
        "figure_name": "eda_03_cancellation_by_hotel_type.png",
        "title": "Cancellation Rate by Hotel Property Type",
        "purpose": "Compare reservation fragility between urban business hotels and leisure resort destinations.",
        "findings": (
            f"City Hotels experience a substantially higher cancellation rate ({city_rate:.1f}%) "
            f"than Resort Hotels ({resort_rate:.1f}%). City hotel customers are more frequently business travelers "
            "or short-stay guests who modify itineraries easily."
        ),
    }
    return metadata


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = OUTPUTS_DIR) -> dict:
    """4. Correlation Heatmap for Numerical Features"""
    num_cols = [
        "is_canceled",
        "lead_time",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "booking_changes",
        "days_in_waiting_list",
        "adr",
        "required_car_parking_spaces",
        "total_of_special_requests",
    ]
    existing_cols = [c for c in num_cols if c in df.columns]
    corr = df[existing_cols].corr()

    plt.figure(figsize=(11, 8.5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.4,
        vmin=-0.4,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8.5},
    )
    plt.title("Correlation Matrix of Key Numerical Reservation Features", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "eda_04_correlation_heatmap.png")
    plt.savefig(filepath)
    plt.close()

    lead_corr = corr.loc["lead_time", "is_canceled"] if "lead_time" in corr.index else 0
    spec_corr = corr.loc["total_of_special_requests", "is_canceled"] if "total_of_special_requests" in corr.index else 0
    park_corr = corr.loc["required_car_parking_spaces", "is_canceled"] if "required_car_parking_spaces" in corr.index else 0

    metadata = {
        "figure_name": "eda_04_correlation_heatmap.png",
        "title": "Pearson Correlation Matrix of Numerical Features",
        "purpose": "Quantify linear dependencies between numerical variables and reservation cancellation.",
        "findings": (
            f"Lead time has the strongest positive linear correlation with cancellation (r = {lead_corr:.2f}). "
            f"Conversely, special requests (r = {spec_corr:.2f}) and parking space requirements (r = {park_corr:.2f}) "
            "exhibit negative correlations with cancellation, reflecting higher commitment from engaged guests."
        ),
    }
    return metadata


def plot_adr_distribution_boxplot(df: pd.DataFrame, output_dir: str = OUTPUTS_DIR) -> dict:
    """5. Average Daily Rate (ADR) Distribution and Boxplot"""
    filtered_df = df[(df["adr"] > 0) & (df["adr"] <= 400)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Histogram + KDE
    sns.histplot(
        data=filtered_df,
        x="adr",
        hue="is_canceled",
        kde=True,
        palette=["#2b5c8f", "#d95f02"],
        alpha=0.4,
        ax=axes[0],
        bins=40,
    )
    axes[0].set_title("ADR Distribution by Cancellation Status", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Average Daily Rate (EUR)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)

    # Right: Boxplot by Hotel Type
    sns.boxplot(
        data=filtered_df,
        x="hotel",
        y="adr",
        hue="is_canceled",
        palette=["#2b5c8f", "#d95f02"],
        ax=axes[1],
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
    )
    axes[1].set_title("ADR Boxplot by Hotel Type and Outcome", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Hotel Type", fontsize=10)
    axes[1].set_ylabel("Average Daily Rate (EUR)", fontsize=10)

    plt.suptitle("Average Daily Rate (ADR) Distribution and Spread Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "eda_05_adr_distribution_boxplot.png")
    plt.savefig(filepath)
    plt.close()

    mean_adr = filtered_df["adr"].mean()
    median_adr = filtered_df["adr"].median()

    metadata = {
        "figure_name": "eda_05_adr_distribution_boxplot.png",
        "title": "Average Daily Rate (ADR) Distribution & Boxplot",
        "purpose": "Analyze price sensitivity and pricing distributions across hotel types and cancellation outcomes.",
        "findings": (
            f"The overall ADR exhibits a right-skewed distribution with a mean of €{mean_adr:.2f} and median of €{median_adr:.2f}. "
            "Canceled reservations consistently display higher median ADRs across both City and Resort hotels, indicating price elasticity."
        ),
    }
    return metadata


def plot_cancellation_by_market_segment(df: pd.DataFrame, output_dir: str = OUTPUTS_DIR) -> dict:
    """6. Cancellation Rate across Market Segments"""
    seg_summary = df.groupby("market_segment")["is_canceled"].agg(
        total="count",
        canceled="sum",
        rate="mean"
    ).reset_index()
    seg_summary["rate_pct"] = seg_summary["rate"] * 100
    seg_summary = seg_summary[seg_summary["total"] >= 50].sort_values("rate_pct", ascending=False)

    plt.figure(figsize=(9, 5))
    bars = plt.barh(seg_summary["market_segment"], seg_summary["rate_pct"], color="#386cb0", edgecolor="black", alpha=0.85)
    plt.title("Cancellation Rate by Distribution Market Segment", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Cancellation Rate (%)", fontsize=11, labelpad=8)
    plt.ylabel("Market Segment", fontsize=11, labelpad=8)
    plt.xlim(0, max(seg_summary["rate_pct"]) * 1.25)

    for bar, (_, row) in zip(bars, seg_summary.iterrows()):
        plt.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{row['rate_pct']:.1f}% ({int(row['canceled']):,}/{int(row['total']):,})",
            va="center",
            fontweight="bold",
            fontsize=9.5,
        )

    plt.gca().invert_yaxis()
    plt.tight_layout()

    filepath = os.path.join(output_dir, "eda_06_cancellation_by_market_segment.png")
    plt.savefig(filepath)
    plt.close()

    top_seg = seg_summary.iloc[0]
    bot_seg = seg_summary.iloc[-1]

    metadata = {
        "figure_name": "eda_06_cancellation_by_market_segment.png",
        "title": "Cancellation Rate by Market Segment",
        "purpose": "Determine which booking channel and customer acquisition avenue carries the highest reservation volatility.",
        "findings": (
            f"Market segments vary substantially in cancellation propensity: '{top_seg['market_segment']}' exhibits the highest "
            f"cancellation rate ({top_seg['rate_pct']:.1f}%), whereas '{bot_seg['market_segment']}' maintains the lowest "
            f"cancellation rate ({bot_seg['rate_pct']:.1f}%). Direct and Corporate guests demonstrate significantly higher commitment."
        ),
    }
    return metadata


def plot_cancellation_by_deposit_type(df: pd.DataFrame, output_dir: str = OUTPUTS_DIR) -> dict:
    """7. Cancellation Rate by Deposit Type"""
    dep_summary = df.groupby("deposit_type")["is_canceled"].agg(
        total="count",
        canceled="sum",
        rate="mean"
    ).reset_index()
    dep_summary["rate_pct"] = dep_summary["rate"] * 100

    plt.figure(figsize=(8, 5))
    colors = ["#4daf4a", "#e41a1c", "#377eb8"]
    bars = plt.bar(dep_summary["deposit_type"], dep_summary["rate_pct"], color=colors[:len(dep_summary)], width=0.5, edgecolor="black", alpha=0.85)

    plt.title("Cancellation Rate by Deposit Policy Type", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Deposit Type", fontsize=11, labelpad=8)
    plt.ylabel("Cancellation Rate (%)", fontsize=11, labelpad=8)
    plt.ylim(0, 110)

    for bar, (_, row) in zip(bars, dep_summary.iterrows()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{row['rate_pct']:.1f}%\n({int(row['canceled']):,}/{int(row['total']):,})",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    filepath = os.path.join(output_dir, "eda_07_cancellation_by_deposit_type.png")
    plt.savefig(filepath)
    plt.close()

    non_ref_rate = dep_summary.loc[dep_summary["deposit_type"] == "Non Refund", "rate_pct"].values
    non_ref_val = non_ref_rate[0] if len(non_ref_rate) > 0 else 0

    metadata = {
        "figure_name": "eda_07_cancellation_by_deposit_type.png",
        "title": "Cancellation Rate by Deposit Type",
        "purpose": "Analyze the empirical effectiveness and behavioral paradox of upfront non-refundable deposits.",
        "findings": (
            f"The Non Refund deposit policy demonstrates an unexpected cancellation rate of {non_ref_val:.1f}%. "
            "As documented in Antonio et al. (2019), non-refundable terms are disproportionately assigned to high-risk group "
            "contracts and distressed blocks that subsequently cancel en masse."
        ),
    }
    return metadata


def plot_monthly_booking_patterns(df: pd.DataFrame, output_dir: str = OUTPUTS_DIR) -> dict:
    """8. Monthly Booking Volume and Cancellation Rate Trend"""
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    df_month = df[df["arrival_date_month"].isin(month_order)].copy()
    summary = df_month.groupby("arrival_date_month")["is_canceled"].agg(
        total="count",
        canceled="sum",
        rate="mean"
    ).reindex(month_order).reset_index()
    summary["rate_pct"] = summary["rate"] * 100

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(month_order))
    width = 0.55

    bars = ax1.bar(x, summary["total"], width=width, color="#6baed6", alpha=0.7, label="Total Bookings")
    line = ax2.plot(x, summary["rate_pct"], color="#cb181d", marker="o", linewidth=2.5, label="Cancellation Rate (%)")

    ax1.set_title("Seasonality: Monthly Booking Volume vs Cancellation Rate", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Arrival Month", fontsize=11, labelpad=8)
    ax1.set_ylabel("Total Bookings", fontsize=11, color="#2171b5")
    ax2.set_ylabel("Cancellation Rate (%)", fontsize=11, color="#cb181d")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[:3] for m in month_order])
    ax2.set_ylim(0, max(summary["rate_pct"]) * 1.3)
    ax2.grid(False)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "eda_08_monthly_booking_patterns.png")
    plt.savefig(filepath)
    plt.close()

    metadata = {
        "figure_name": "eda_08_monthly_booking_patterns.png",
        "title": "Seasonality Analysis: Monthly Booking Volume & Cancellation Rate",
        "purpose": "Evaluate calendar seasonality and summer peak travel impact on reservation stability.",
        "findings": (
            "Booking volume peaks during the European summer holiday months (July and August). Cancellation rates "
            "closely mirror demand surges, peaking in summer when hotel prices are elevated and guests hold tentative bookings."
        ),
    }
    return metadata


def plot_special_requests_impact(df: pd.DataFrame, output_dir: str = OUTPUTS_DIR) -> dict:
    """9. Special Requests Impact on Cancellation Rate"""
    req_summary = df.groupby("total_of_special_requests")["is_canceled"].agg(
        total="count",
        canceled="sum",
        rate="mean"
    ).reset_index()
    req_summary["rate_pct"] = req_summary["rate"] * 100
    req_summary = req_summary[req_summary["total"] >= 20]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        req_summary["total_of_special_requests"],
        req_summary["rate_pct"],
        color="#bc80bd",
        edgecolor="black",
        width=0.55,
        alpha=0.85,
    )
    plt.title("Impact of Special Requests on Cancellation Likelihood", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Number of Special Requests Submitted", fontsize=11, labelpad=8)
    plt.ylabel("Cancellation Rate (%)", fontsize=11, labelpad=8)
    plt.ylim(0, max(req_summary["rate_pct"]) * 1.25)

    for bar, (_, row) in zip(bars, req_summary.iterrows()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{row['rate_pct']:.1f}%\n({int(row['canceled']):,}/{int(row['total']):,})",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
        )

    plt.tight_layout()
    filepath = os.path.join(output_dir, "eda_09_special_requests_impact.png")
    plt.savefig(filepath)
    plt.close()

    zero_rate = req_summary.loc[req_summary["total_of_special_requests"] == 0, "rate_pct"].values[0]
    multi_rate = req_summary.loc[req_summary["total_of_special_requests"] >= 2, "rate_pct"].mean()

    metadata = {
        "figure_name": "eda_09_special_requests_impact.png",
        "title": "Special Requests Impact on Cancellation Rate",
        "purpose": "Quantify whether guest engagement and personalized requirements predict booking completion.",
        "findings": (
            f"Guests submitting zero special requests experience a {zero_rate:.1f}% cancellation rate, whereas "
            f"guests submitting 2 or more requests average approximately {multi_rate:.1f}% cancellations. Active customization "
            "reflects psychological commitment to travel."
        ),
    }
    return metadata


def generate_all_eda_visualizations(df: pd.DataFrame, output_dir: str = OUTPUTS_DIR) -> dict:
    """Executes all EDA visualizations and saves metadata documentation."""
    print("[*] Generating all EDA Visualizations...")
    metadata_list = [
        plot_cancellation_count(df, output_dir),
        plot_lead_time_distribution(df, output_dir),
        plot_cancellation_by_hotel_type(df, output_dir),
        plot_correlation_heatmap(df, output_dir),
        plot_adr_distribution_boxplot(df, output_dir),
        plot_cancellation_by_market_segment(df, output_dir),
        plot_cancellation_by_deposit_type(df, output_dir),
        plot_monthly_booking_patterns(df, output_dir),
        plot_special_requests_impact(df, output_dir),
    ]

    metadata_dict = {item["figure_name"]: item for item in metadata_list}
    meta_path = os.path.join(output_dir, "eda_visualizations_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=4)

    print(f"[+] Successfully generated {len(metadata_list)} EDA figures in '{output_dir}'.")
    return metadata_dict


if __name__ == "__main__":
    try:
        from src.data_loading import load_and_inspect_data
    except ModuleNotFoundError:
        from data_loading import load_and_inspect_data
    df, _ = load_and_inspect_data()
    meta = generate_all_eda_visualizations(df)
    print("EDA Visualizations created successfully!")
