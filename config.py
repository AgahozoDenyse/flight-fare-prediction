# config.py — single source of truth for all pipeline constants

RAW_DATA_PATH  = "./data/Flight_Price_Dataset_of_Bangladesh.csv"
RANDOM_STATE   = 42
TARGET_COLUMN  = "Total Fare (BDT)"

# Both components sum to reconstruct the target — keeping either causes leakage
DROP_COLUMNS = [
    "Base Fare (BDT)",
    "Tax & Surcharge (BDT)",
]

# Columns to one-hot encode during preprocessing
CATEGORICAL_COLUMNS = [
    "Airline",
    "Source",
    "Destination",
    "Stopovers",
    "Class",
]

# Raw numeric columns present in the CSV (engineered columns like Dep_Hour are excluded)
NUMERIC_COLUMNS = [
    "Duration (hrs)",
    "Days Before Departure",
]

# ── Validation schemas ─────────────────────────────────────────────────────────

# Expected dtypes for the raw CSV used by data_loader to flag schema drift.
# Dates arrive as plain strings from CSV — dtype is "object" until preprocessing parses them.
EXPECTED_COLUMN_TYPES = {
    "Airline":               "object",
    "Source":                "object",
    "Source Name":           "object",
    "Destination":           "object",
    "Destination Name":      "object",
    "Departure Date & Time": "object",
    "Arrival Date & Time":   "object",
    "Duration (hrs)":        "float64",
    "Stopovers":             "object",
    "Aircraft Type":         "object",
    "Class":                 "object",
    "Booking Source":        "object",
    "Base Fare (BDT)":       "float64",
    "Tax & Surcharge (BDT)": "float64",
    "Total Fare (BDT)":      "float64",
    "Seasonality":           "object",
    "Days Before Departure":  "int64",
}

# Expected dtypes just before preprocessing transforms (raw strings, not yet parsed)
EXPECTED_SCHEMA_BEFORE = {
    "Departure Date & Time": "object",
    "Arrival Date & Time":   "object",
    "Duration (hrs)":        "float64",
    "Stopovers":             "object",
    "Base Fare (BDT)":       "float64",
    "Tax & Surcharge (BDT)": "float64",
    "Total Fare (BDT)":      "float64",
    "Days Before Departure":  "int64",
}

# Expected dtypes after datetime columns have been parsed and new columns created
EXPECTED_SCHEMA_AFTER = {
    "Dep_DateTime":     "datetime64",
    "Arrival_DateTime": "datetime64",
    "Dep_Hour":         "int64",
    "Arrival_Hour":     "int64",
    "Duration_mins":    "float64",
}

# ── Imputation strategies ──────────────────────────────────────────────────────

# Used by data_loader to document intent; applied in preprocessing._impute_missing_values
IMPUTATION_STRATEGY = {
    "Duration (hrs)":        "median",
    "Days Before Departure":  "median",
    "Stopovers":             "mode",
    "Seasonality":           "mode",
    "Departure Date & Time": "drop_row",
    "Arrival Date & Time":   "drop_row",
}
