import logging
import calendar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from config import TARGET_COLUMN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

PLOT_DIR = Path("plots")
FARE_COL = TARGET_COLUMN


def run_eda(df: pd.DataFrame) -> None:
    """Generate and save all EDA plots and log KPI summaries."""
    PLOT_DIR.mkdir(exist_ok=True)

    _descriptive_stats(df)
    _plot_fare_distribution(df)
    _plot_avg_fare_by_airline(df)
    _plot_fare_by_season(df)
    _plot_fare_by_month(df)
    _plot_top_routes(df)
    _plot_correlation_heatmap(df)
    _log_kpis(df)

    logger.info("EDA complete — plots saved to %s/", PLOT_DIR)


def _descriptive_stats(df: pd.DataFrame) -> None:
    """Log numeric and categorical descriptive statistics."""
    logger.info("=== Descriptive Statistics ===")
    logger.info("Shape: %d rows × %d columns", *df.shape)
    logger.info("Numeric summary:\n%s", df.describe().T.to_string())
    if df.select_dtypes(include="object").shape[1]:
        logger.info("Categorical summary:\n%s", df.describe(include="object").T.to_string())


def _plot_fare_distribution(df: pd.DataFrame) -> None:
    """Plot raw and log-transformed Total Fare distributions side by side."""
    if FARE_COL not in df.columns:
        logger.warning("%s not found — skipping fare distribution plot.", FARE_COL)
        return

    fares = df[FARE_COL].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    fares.plot.hist(bins=50, ax=axes[0], edgecolor="white", color="steelblue")
    axes[0].set_title("Total Fare Distribution")
    axes[0].set_xlabel("Total Fare (BDT)")
    axes[0].set_ylabel("Count")

    np.log1p(fares).plot.hist(bins=50, ax=axes[1], edgecolor="white", color="coral")
    axes[1].set_title("Log-Transformed Total Fare")
    axes[1].set_xlabel("log1p(Total Fare)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "fare_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: fare_distribution.png")


def _plot_avg_fare_by_airline(df: pd.DataFrame) -> None:
    """Plot average Total Fare per airline as a horizontal bar chart."""
    if "Airline" not in df.columns or FARE_COL not in df.columns:
        logger.warning("Airline or %s not found — skipping.", FARE_COL)
        return

    avg = df.groupby("Airline")[FARE_COL].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, max(4, len(avg) * 0.45)))
    avg.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Average Total Fare by Airline")
    ax.set_xlabel("Average Fare (BDT)")
    ax.set_ylabel("Airline")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "avg_fare_by_airline.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: avg_fare_by_airline.png")


def _plot_fare_by_season(df: pd.DataFrame) -> None:
    """Plot fare variation across seasons as a boxplot ordered by median fare."""
    season_col = next((c for c in ["Seasonality", "Season"] if c in df.columns), None)
    if season_col is None or FARE_COL not in df.columns:
        logger.warning("Season column or %s not found — skipping.", FARE_COL)
        return

    order = df.groupby(season_col)[FARE_COL].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x=season_col, y=FARE_COL, order=order, ax=ax)
    ax.set_title("Fare Variation Across Seasons")
    ax.set_xlabel("Season")
    ax.set_ylabel("Total Fare (BDT)")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "fare_by_season.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: fare_by_season.png")


def _plot_fare_by_month(df: pd.DataFrame) -> None:
    """Plot average fare by departure month."""
    date_col = next(
        (c for c in ["Departure Date & Time", "Date_of_Journey"] if c in df.columns), None
    )
    if date_col is None or FARE_COL not in df.columns:
        logger.warning("Date column or %s not found — skipping.", FARE_COL)
        return

    months = pd.to_datetime(df[date_col], errors="coerce").dt.month
    monthly_avg = df[FARE_COL].groupby(months).mean().rename_axis("Month")

    fig, ax = plt.subplots(figsize=(10, 4))
    monthly_avg.plot.bar(ax=ax, color="teal", edgecolor="white", rot=0)
    ax.set_title("Average Fare by Departure Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Fare (BDT)")
    ax.set_xticklabels(
        [calendar.month_abbr[int(m)] for m in monthly_avg.index], rotation=45
    )

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "avg_fare_by_month.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: avg_fare_by_month.png")


def _plot_top_routes(df: pd.DataFrame) -> None:
    """Plot top 10 most popular routes and top 5 most expensive routes."""
    if "Source" not in df.columns or "Destination" not in df.columns:
        logger.warning("Source or Destination not found — skipping route plots.")
        return

    df = df.copy()
    df["Route"] = df["Source"] + " → " + df["Destination"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df["Route"].value_counts().head(10).sort_values().plot.barh(
        ax=axes[0], color="steelblue", edgecolor="white"
    )
    axes[0].set_title("Top 10 Most Popular Routes")
    axes[0].set_xlabel("Number of Flights")

    if FARE_COL in df.columns:
        df.groupby("Route")[FARE_COL].mean().nlargest(5).sort_values().plot.barh(
            ax=axes[1], color="coral", edgecolor="white"
        )
        axes[1].set_title("Top 5 Most Expensive Routes (Avg Fare)")
        axes[1].set_xlabel("Average Fare (BDT)")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "routes.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: routes.png")


def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation heatmap for numeric features, excluding leakage columns."""
    leakage = {"Base Fare (BDT)", "Tax & Surcharge (BDT)"}
    numeric_df = df.select_dtypes(include="number").drop(
        columns=[c for c in leakage if c in df.columns]
    )

    if numeric_df.shape[1] < 2:
        logger.warning("Not enough numeric columns for correlation heatmap.")
        return

    corr = numeric_df.corr()
    size = max(8, len(corr) * 0.6)

    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.4, ax=ax, annot_kws={"size": 8}
    )
    ax.set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: correlation_heatmap.png")


def _log_kpis(df: pd.DataFrame) -> None:
    """Log KPIs: avg fare by airline, most popular route, seasonal variation, top 5 routes."""
    logger.info("=== KPIs ===")

    if "Airline" in df.columns and FARE_COL in df.columns:
        logger.info(
            "Average Fare by Airline:\n%s",
            df.groupby("Airline")[FARE_COL].mean().sort_values(ascending=False).to_string()
        )

    if "Source" in df.columns and "Destination" in df.columns:
        route_counts = (df["Source"] + " → " + df["Destination"]).value_counts()
        logger.info("Most Popular Route: %s (%d flights)", route_counts.index[0], route_counts.iloc[0])

        if FARE_COL in df.columns:
            df_copy = df.copy()
            df_copy["Route"] = df_copy["Source"] + " → " + df_copy["Destination"]
            top5 = df_copy.groupby("Route")[FARE_COL].mean().nlargest(5)
            logger.info("Top 5 Most Expensive Routes:\n%s", top5.to_string())

    season_col = next((c for c in ["Seasonality", "Season"] if c in df.columns), None)
    if season_col and FARE_COL in df.columns:
        logger.info(
            "Average Fare by Season:\n%s",
            df.groupby(season_col)[FARE_COL].mean().sort_values(ascending=False).to_string()
        )


if __name__ == "__main__":
    from data_loader import load_data
    raw_df = load_data("data/Flight_Price_Dataset_of_Bangladesh.csv")
    run_eda(raw_df)
