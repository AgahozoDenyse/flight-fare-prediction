# preprocessing.py
import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from config import (
    DROP_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN,
    EXPECTED_SCHEMA_BEFORE, EXPECTED_SCHEMA_AFTER,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: datetime parsing, feature extraction, imputation, encoding."""
    df = df.copy()

    _validate_schema(df, EXPECTED_SCHEMA_BEFORE, "BEFORE TRANSFORMS")

    df = _convert_datetime(df)
    df = _extract_time_features(df)

    _validate_schema(df, EXPECTED_SCHEMA_AFTER, "AFTER TRANSFORMS")

    df = _impute_missing_values(df)
    df = _drop_leakage_columns(df)
    df = _add_route_popularity(df)   # must run before OHE removes Source/Destination
    df = _encode_categorical(df)

    logger.info("Preprocessing completed — shape: %s", df.shape)
    return df

def _validate_schema(df: pd.DataFrame, schema: dict, label: str) -> None:
    """Validate actual dtypes against expected schema."""
    logger.info("--- Data Type Validation: %s ---", label)
    for col, expected_type in schema.items():
        actual_type = str(df[col].dtype) if col in df.columns else "NOT FOUND"
        # Strip bit-width suffix (e.g. "int64" → "int") so "Int64" and "int64" both match "int64"
        status = "OK" if actual_type.startswith(expected_type.replace("64", "")) else " MISMATCH"
        logger.info("  %-26s actual=%-14s expected=%-14s %s", col, actual_type, expected_type, status)

def _convert_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "Departure Date & Time" in df.columns:
        df["Dep_DateTime"] = pd.to_datetime(df["Departure Date & Time"], errors="coerce")
        logger.info("  Dep_DateTime parsed — NaT: %d", df["Dep_DateTime"].isna().sum())
    if "Arrival Date & Time" in df.columns:
        df["Arrival_DateTime"] = pd.to_datetime(df["Arrival Date & Time"], errors="coerce")
        logger.info("  Arrival_DateTime parsed — NaT: %d", df["Arrival_DateTime"].isna().sum())
    return df

def _extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Dep_DateTime" in df.columns:
        df["Dep_Hour"] = df["Dep_DateTime"].dt.hour.astype("int64")
        logger.info("  Dep_Hour extracted — range: %d–%d", df["Dep_Hour"].min(), df["Dep_Hour"].max())
    if "Arrival_DateTime" in df.columns:
        df["Arrival_Hour"] = df["Arrival_DateTime"].dt.hour.astype("int64")
        logger.info("  Arrival_Hour extracted — range: %d–%d", df["Arrival_Hour"].min(), df["Arrival_Hour"].max())
    if "Duration (hrs)" in df.columns:
        df["Duration_mins"] = (df["Duration (hrs)"] * 60).round(2)
        logger.info("  Duration_mins created — range: %.1f–%.1f", df["Duration_mins"].min(), df["Duration_mins"].max())
    return df

def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_before = df.isnull().sum()[lambda x: x > 0]
    if not missing_before.empty:
        logger.info("--- Missing Value Audit (before imputation) ---")
        for col, cnt in missing_before.items():
            pct = cnt / len(df) * 100
            logger.info("  %-28s %4d missing (%5.2f%%)", col, cnt, pct)

    imputation_log = {}

    # Also catch engineered columns (Duration_mins, Dep_Hour, Arrival_Hour) not listed in NUMERIC_COLUMNS
    numeric_candidates = [c for c in df.columns if c in NUMERIC_COLUMNS or c.endswith("_mins") or c.endswith("_Hour")]
    for col in numeric_candidates:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        imputation_log[col] = {"strategy": "median", "value": round(float(median_val),2), "n": n_missing}

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        df[col] = df[col].fillna("Unknown")
        imputation_log[col] = {"strategy": "constant='Unknown'", "value": "Unknown", "n": n_missing}

    date_col = "Departure Date & Time"
    if date_col in df.columns:
        n_before = len(df)
        df = df.dropna(subset=[date_col]).copy()
        logger.info("  Dropped %d rows — missing %s", n_before - len(df), date_col)

    if TARGET_COLUMN in df.columns:
        n_before = len(df)
        df = df[df[TARGET_COLUMN] > 0].copy()
        logger.info("  Dropped %d rows — %s <= 0", n_before - len(df), TARGET_COLUMN)

    if imputation_log:
        logger.info("--- Imputation Summary (after filling) ---")
        for col, info in imputation_log.items():
            logger.info("  %-28s strategy=%-22s value=%-12s n_filled=%d",
                        col, info["strategy"], str(info["value"]), info["n"])

    return df

def _add_route_popularity(df: pd.DataFrame) -> pd.DataFrame:
    if "Source" not in df.columns or "Destination" not in df.columns:
        return df
    counts = (
        df.groupby(["Source", "Destination"])
          .size()
          .reset_index(name="Route_Popularity")
    )
    df = df.merge(counts, on=["Source", "Destination"], how="left")
    logger.info("  Route_Popularity added — range: %d–%d",
                df["Route_Popularity"].min(), df["Route_Popularity"].max())
    return df


def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    dropped = []
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=col)
            dropped.append(col)
    logger.info("Leakage columns dropped: %s", dropped)
    return df

def _encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols_present = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    if not cat_cols_present:
        logger.info("No categorical columns found to encode.")
        return df
    # drop='first' removes one dummy per category to avoid the dummy variable trap in linear models
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded = encoder.fit_transform(df[cat_cols_present])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols_present), index=df.index)
    df = df.drop(columns=cat_cols_present)
    df = pd.concat([df, encoded_df], axis=1)
    logger.info("OHE applied — %d categorical columns → %d binary features", len(cat_cols_present), encoded_df.shape[1])
    return df

if __name__ == "__main__":
    from data_loader import load_data
    raw_df = load_data("data/Flight_Price_Dataset_of_Bangladesh.csv")
    clean_df = preprocess_data(raw_df)
    print(f"\nShape: {clean_df.shape}")
    print(clean_df.dtypes)