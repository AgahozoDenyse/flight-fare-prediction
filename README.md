# Flight Fare Prediction

An end-to-end machine learning pipeline that predicts flight fares for Bangladesh routes. The project covers the full ML lifecycle and is orchestrated with Apache Airflow 3.2 via Docker.

## Overview

This project builds a regression pipeline on the Bangladesh Flight Price Dataset to predict `Total Fare (BDT)`. It trains and compares five models, selects the best performer, and generates automated insights. The pipeline runs on a weekly schedule via Airflow, with a separate monthly retraining DAG.

## Project Structure

```
├── config.py                        # Centralized configuration
├── data_loader.py                   # Data loading & validation
├── eda.py                           # Exploratory data analysis
├── preprocessing.py                 # Cleaning, datetime parsing, encoding
├── feature_engineering.py           # Temporal features, route popularity
├── modeling.py                      # Model training, evaluation, selection
├── insights.py                      # Feature importance, bias-variance, report
├── run_pipeline.py                  # Local single-script runner
├── docker-compose.airflow.yaml      # Airflow 3.2 stack (Docker)
├── dags/
│   ├── flight_fare_pipeline.py      # Weekly end-to-end pipeline DAG
│   └── flight_fare_retrain.py       # Monthly retraining DAG
├── data/
│   └── Flight_Price_Dataset_of_Bangladesh.csv
├── models/                          # Saved models & metadata
├── plots/                           # Generated visualizations
└── reports/                         # Summary report
```

## Pipeline Steps

```
load_and_validate
    ├── run_eda          (parallel)
    └── preprocess
            └── feature_engineering
                    └── modeling
                            └── insights
```

## Models Trained

| Model | Tuning |
|-------|--------|
| Linear Regression | — |
| Ridge | GridSearchCV |
| Lasso | GridSearchCV |
| Decision Tree | GridSearchCV |
| Random Forest | GridSearchCV |

The best model is saved to `models/best_model.joblib` with metadata in `models/best_model_info.json`.

## Key Findings

- **Travel Class** accounts for ~45% of fare variation
- **Flight Duration** accounts for ~42%
- **Destination** accounts for ~10%

## Setup & Running

### Option 1 — Run locally

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib pyarrow
python run_pipeline.py
```

### Option 2 — Run with Airflow (Docker)

**Prerequisites:** Docker Desktop

```bash
# Start the stack
docker compose -f docker-compose.airflow.yaml up -d

# Open the Airflow UI
# http://localhost:8080

# Trigger the pipeline DAG from the UI
# DAG: flight_fare_pipeline
```

Wait for the webserver to show `(healthy)` (~2 minutes), then trigger `flight_fare_pipeline` from the Airflow UI.

### Stop the stack

```bash
docker compose -f docker-compose.airflow.yaml down
```

## Outputs

| Output | Location |
|--------|----------|
| Model files | `models/*.joblib` |
| Model comparison | `models/comparison.csv` |
| Best model info | `models/best_model_info.json` |
| EDA plots | `plots/` |
| Feature importance | `plots/feature_importance_grouped.png` |
| Bias-variance plot | `plots/bias_variance_tradeoff.png` |
| Summary report | `reports/summary.md` |

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- **Apache Airflow 3.2** — pipeline orchestration
- **Docker** — containerized Airflow stack
- **PostgreSQL** — Airflow metadata database
