"""
Retraining DAG: runs monthly to retrain the best saved model on the latest data.

Does NOT re-run EDA or full modeling — only retrains the winning model
(identified in models/best_model_info.json) using retrain_best_model().
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
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


@dag(
    dag_id="flight_fare_retrain",
    description="Monthly retraining of the best model on the latest data",
    schedule="@monthly",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    default_args=default_args,
    tags=["ml", "flight-fare", "retrain"],
)
def flight_fare_retrain():

    @task()
    def load_latest_data() -> str:
        """Load the latest data file and save as Parquet."""
        _add_project_to_path()
        import pandas as pd
        from data_loader import load_data

        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        df  = load_data(str(PROJECT_DIR / "data" / "Flight_Price_Dataset_of_Bangladesh.csv"))
        out = str(INTERIM_DIR / "retrain_raw.parquet")
        df.to_parquet(out, index=False)
        return out

    @task()
    def prepare_features(raw_path: str) -> str:
        """Preprocess + feature-engineer the new data."""
        _add_project_to_path()
        import pandas as pd
        from preprocessing import preprocess_data
        from feature_engineering import create_features

        df          = pd.read_parquet(raw_path)
        clean_df    = preprocess_data(df)
        featured_df = create_features(clean_df)
        out         = str(INTERIM_DIR / "retrain_featured.parquet")
        featured_df.to_parquet(out, index=False)
        return out

    @task()
    def retrain(featured_path: str) -> None:
        """Retrain the saved best model and overwrite models/best_model.joblib."""
        _add_project_to_path()
        import pandas as pd
        from modeling import retrain_best_model

        df = pd.read_parquet(featured_path)
        retrain_best_model(df)

    # ── DAG wiring ─────────────────────────────────────────────────────────────
    raw_path      = load_latest_data()
    featured_path = prepare_features(raw_path)
    retrain(featured_path)


flight_fare_retrain()
