"""
Main pipeline DAG: load → EDA → preprocess → feature engineering → modeling → insights.

EDA runs in parallel with preprocessing since both only need the raw data.
Tasks share data via Parquet files written to data/interim/ — XCom carries only paths.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

PROJECT_DIR = Path("/opt/airflow/project")
INTERIM_DIR = PROJECT_DIR / "data" / "interim"


def _add_project_to_path() -> None:
    project = str(PROJECT_DIR)
    if project not in sys.path:
        sys.path.insert(0, project)


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="flight_fare_pipeline",
    description="End-to-end flight fare prediction pipeline",
    schedule="@weekly",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    default_args=default_args,
    tags=["ml", "flight-fare"],
)
def flight_fare_pipeline():

    @task()
    def load_and_validate() -> str:
        """Load raw CSV, run validation checks, save to Parquet."""
        _add_project_to_path()
        import pandas as pd
        from data_loader import load_data

        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        df = load_data(str(PROJECT_DIR / "data" / "Flight_Price_Dataset_of_Bangladesh.csv"))
        out = str(INTERIM_DIR / "raw.parquet")
        df.to_parquet(out, index=False)
        return out

    @task()
    def run_eda(raw_path: str) -> None:
        """Run EDA on raw data — runs in parallel with preprocessing."""
        _add_project_to_path()
        import pandas as pd
        from eda import run_eda as _run_eda

        df = pd.read_parquet(raw_path)
        _run_eda(df)

    @task()
    def preprocess(raw_path: str) -> str:
        """Clean raw data: datetime parsing, imputation, leakage drop, OHE."""
        _add_project_to_path()
        import pandas as pd
        from preprocessing import preprocess_data

        df       = pd.read_parquet(raw_path)
        clean_df = preprocess_data(df)
        out      = str(INTERIM_DIR / "clean.parquet")
        clean_df.to_parquet(out, index=False)
        return out

    @task()
    def feature_engineering(clean_path: str) -> str:
        """Add temporal features, route popularity, peak season flag, log target."""
        _add_project_to_path()
        import pandas as pd
        from feature_engineering import create_features

        df          = pd.read_parquet(clean_path)
        featured_df = create_features(df)
        out         = str(INTERIM_DIR / "featured.parquet")
        featured_df.to_parquet(out, index=False)
        return out

    @task()
    def modeling(featured_path: str) -> str:
        """Train all models, tune with GridSearchCV, save each and select the best."""
        _add_project_to_path()
        import pandas as pd
        from modeling import run_modeling

        df         = pd.read_parquet(featured_path)
        comparison = run_modeling(df)
        out        = str(INTERIM_DIR / "comparison.parquet")
        comparison.to_parquet(out)
        return out

    @task()
    def insights(raw_path: str, featured_path: str, comparison_path: str) -> None:
        """Generate feature importance analysis, bias-variance plot, and summary report."""
        _add_project_to_path()
        import pandas as pd
        from insights import run_insights

        raw_df      = pd.read_parquet(raw_path)
        featured_df = pd.read_parquet(featured_path)
        comparison  = pd.read_parquet(comparison_path)
        run_insights(raw_df, featured_df, comparison)

    # ── DAG wiring ─────────────────────────────────────────────────────────────
    raw_path        = load_and_validate()
    run_eda(raw_path)                           # parallel branch — EDA only
    clean_path      = preprocess(raw_path)
    featured_path   = feature_engineering(clean_path)
    comparison_path = modeling(featured_path)
    insights(raw_path, featured_path, comparison_path)


flight_fare_pipeline()
