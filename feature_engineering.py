# feature_engineering.py
import logging
import numpy as np
import pandas as pd
from config import TARGET_COLUMN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

PEAK_SEASON_VALUES = {"Winter Holidays", "Eid Season", "Summer Peak"}


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all derived features: temporal, route popularity, peak season flag, log target."""
    df = df.copy()

    df = _add_temporal_features(df)
    df = _add_route_popularity(df)
    df = _add_peak_season_flag(df)
    df = _add_log_target(df)

    logger.info("create_features() complete — shape: %d rows × %d columns", *df.shape)
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract month, day, weekday, hour, and weekend flag from departure datetime."""
    if "Dep_DateTime" not in df.columns:
        logger.warning("Dep_DateTime not found — temporal features skipped.")
        return df

    df["Departure_Month"]   = df["Dep_DateTime"].dt.month
    df["Departure_Day"]     = df["Dep_DateTime"].dt.day
    df["Departure_Weekday"] = df["Dep_DateTime"].dt.dayofweek  # 0 = Monday
    df["Departure_Hour"]    = df["Dep_DateTime"].dt.hour
    df["Is_Weekend"]        = (df["Departure_Weekday"] >= 5).astype(int)

    logger.info("  Temporal features added: Departure_Month, Departure_Day, "
                "Departure_Weekday, Departure_Hour, Is_Weekend")
    return df


def _add_route_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """Add flight count per source→destination pair as a proxy for market competition."""
    if "Route_Popularity" in df.columns:
        logger.info("  Route_Popularity already present — skipped.")
        return df
    if "Source" not in df.columns or "Destination" not in df.columns:
        logger.warning("Source or Destination column not found — route popularity skipped.")
        return df

    route_counts = (
        df.groupby(["Source", "Destination"])
          .size()
          .reset_index(name="Route_Popularity")
    )
    df = df.merge(route_counts, on=["Source", "Destination"], how="left")

    logger.info("  Route_Popularity added — range: %d–%d",
                df["Route_Popularity"].min(), df["Route_Popularity"].max())
    return df


def _add_peak_season_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary Is_Peak_Season flag alongside OHE Seasonality for linear models."""
    if "Seasonality" not in df.columns:
        logger.warning("Seasonality column not found — peak season flag skipped.")
        return df

    df["Is_Peak_Season"] = df["Seasonality"].isin(PEAK_SEASON_VALUES).astype(int)

    peak_count  = df["Is_Peak_Season"].sum()
    peak_pct    = peak_count / len(df) * 100
    logger.info("  Is_Peak_Season added — %d peak rows (%.1f%%)",
                peak_count, peak_pct)
    return df


def _add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to the fare target to reduce right skew before training."""
    if TARGET_COLUMN not in df.columns:
        logger.warning("%s not found — log target skipped.", TARGET_COLUMN)
        return df

    df["Log_Total_Fare"] = np.log1p(df[TARGET_COLUMN])  # log1p handles zero fares; invert with expm1()

    raw_skewness = df[TARGET_COLUMN].skew()
    log_skewness = df["Log_Total_Fare"].skew()
    logger.info("  Log_Total_Fare added — skewness: %.4f → %.4f  (log1p justified)",
                raw_skewness, log_skewness)
    return df


if __name__ == "__main__":
    from data_loader    import load_data
    from preprocessing  import preprocess_data

    raw_df      = load_data("data/Flight_Price_Dataset_of_Bangladesh.csv")
    clean_df    = preprocess_data(raw_df)
    featured_df = create_features(clean_df)

    print(f"\nShape: {featured_df.shape}")
    print(featured_df[[
        "Departure_Month", "Departure_Weekday", "Departure_Hour",
        "Is_Weekend", "Route_Popularity", "Is_Peak_Season", "Log_Total_Fare"
    ]].head())