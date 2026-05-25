import logging
import pandas as pd
from config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    DROP_COLUMNS,
    EXPECTED_COLUMN_TYPES,
    IMPUTATION_STRATEGY,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV and log structure, dtype mismatches, missing values, and leakage."""
    raw_df = pd.read_csv(path)
    logger.info("Dataset loaded — %d rows × %d columns", *raw_df.shape)

    _log_structure(raw_df)
    _validate_column_types(raw_df)
    _audit_missing_values(raw_df)
    _detect_outliers_and_leakage(raw_df)

    return raw_df


def _log_structure(df: pd.DataFrame) -> None:
    logger.info("Columns: %s", df.columns.tolist())
    logger.info("Numeric summary:\n%s", df.describe().T.to_string())
    logger.info("Categorical summary:\n%s", df.describe(include="object").T.to_string())
    logger.info("First 3 rows:\n%s", df.head(3).to_string())


def _validate_column_types(df: pd.DataFrame) -> None:
    """Compare actual dtypes against the expected schema from config and log mismatches."""
    logger.info("--- Data Type Validation (actual vs expected) ---")
    for column, expected_type in EXPECTED_COLUMN_TYPES.items():
        actual_type = str(df[column].dtype) if column in df.columns else "NOT FOUND"
        type_matches = actual_type.startswith(
            expected_type.replace("64", "").replace("32", "")
        )
        status = " OK" if type_matches else "MISMATCH"
        logger.info("  %-26s actual=%-14s expected=%-14s %s",
                    column, actual_type, expected_type, status)


def _audit_missing_values(df: pd.DataFrame) -> None:
    """Log columns with missing values, their count, percentage, and planned imputation."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        logger.info("No missing values found.")
        return

    logger.info("--- Missing Value Audit ---")
    for column, count in missing.items():
        pct      = count / len(df) * 100
        strategy = IMPUTATION_STRATEGY.get(column, "review required")
        logger.info("  %-26s %4d missing (%5.2f%%)  strategy: %s",
                    column, count, pct, strategy)


def _detect_outliers_and_leakage(df: pd.DataFrame) -> None:
    """Detect outliers via 1.5×IQR on fare columns and flag target leakage."""
    fare_cols = [TARGET_COLUMN] + [c for c in DROP_COLUMNS if c in df.columns]
    logger.info("--- Outlier Detection (1.5 × IQR) ---")

    for col in fare_cols:
        if col not in df.columns:
            continue
        values      = df[col].dropna()
        q1, q3      = values.quantile(0.25), values.quantile(0.75)
        iqr         = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_low       = (values < lower).sum()
        n_high      = (values > upper).sum()
        logger.info("  %-26s fence=[%8.0f, %8.0f]  outliers: %d low | %d high",
                    col, lower, upper, n_low, n_high)

    logger.info("  Retain high-fare outliers; drop rows where %s <= 0 in preprocessing.py.",
                TARGET_COLUMN)

    # Leakage check: if DROP_COLUMNS sum reconstructs the target almost perfectly, they must be dropped
    if all(c in df.columns for c in DROP_COLUMNS) and TARGET_COLUMN in df.columns:
        reconstructed = sum(df[c] for c in DROP_COLUMNS)
        corr = reconstructed.corr(df[TARGET_COLUMN])
        logger.info("--- Target Leakage Check ---")
        logger.info("  Correlation (%s) vs %s: %.6f",
                    " + ".join(DROP_COLUMNS), TARGET_COLUMN, corr)
        if corr > 0.99:
            logger.warning("  LEAKAGE RISK: %s reconstructs %s (r=%.4f)",
                           " + ".join(DROP_COLUMNS), TARGET_COLUMN, corr)


if __name__ == "__main__":
    df = load_data(RAW_DATA_PATH)
    print("Data loaded successfully! Shape:", df.shape)
    print(df.head())
