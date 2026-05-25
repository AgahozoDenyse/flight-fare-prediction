import json
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from config import RANDOM_STATE, TARGET_COLUMN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

PLOT_DIR   = Path("plots")
MODELS_DIR = Path("models")

_DROP_ALWAYS = {
    TARGET_COLUMN, "Log_Total_Fare",
    "Dep_DateTime", "Arrival_DateTime",
    "Departure Date & Time", "Arrival Date & Time",
}

PARAM_GRIDS = {
    "Ridge":        {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso":        {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "DecisionTree": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
}


# ── Public entry point ─────────────────────────────────────────────────────────

def run_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Train all models separately, compare outputs, save each, and persist the best."""
    PLOT_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names, scaler = _prepare_data(df)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    logger.info("Saved scaler → models/scaler.joblib")

    trained = {}
    results = {}

    trained["LinearRegression"], results["LinearRegression"] = train_linear_regression(
        X_train, X_test, y_train, y_test, feature_names)

    trained["Ridge"], results["Ridge"] = train_ridge(
        X_train, X_test, y_train, y_test, feature_names)

    trained["Lasso"], results["Lasso"] = train_lasso(
        X_train, X_test, y_train, y_test, feature_names)

    trained["DecisionTree"], results["DecisionTree"] = train_decision_tree(
        X_train, X_test, y_train, y_test, feature_names)

    trained["RandomForest"], results["RandomForest"] = train_random_forest(
        X_train, X_test, y_train, y_test, feature_names)

    comparison = _comparison_table(results)
    _plot_model_comparison(comparison)
    comparison.to_csv(MODELS_DIR / "comparison.csv")
    logger.info("Saved comparison table → models/comparison.csv")

    best_name = str(comparison["R²"].idxmax())
    _save_best_model(trained[best_name], best_name, results[best_name], feature_names)

    return comparison


# ── Per-model training functions ───────────────────────────────────────────────

def train_linear_regression(X_train, X_test, y_train, y_test, feature_names) -> tuple:
    logger.info("=== Linear Regression (Baseline) ===")
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    metrics = _record_metrics("LinearRegression", y_test, y_pred)

    _plot_actual_vs_predicted(y_test, y_pred, "LinearRegression")
    _plot_residuals(y_test, y_pred, "LinearRegression")
    _plot_feature_importance(model, feature_names, "LinearRegression")
    _save_model(model, "LinearRegression")
    return model, metrics


def train_ridge(X_train, X_test, y_train, y_test, feature_names) -> tuple:
    logger.info("=== Ridge Regression ===")
    model  = _tune_and_validate("Ridge", Ridge(max_iter=10000), X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = _record_metrics("Ridge", y_test, y_pred)

    _plot_actual_vs_predicted(y_test, y_pred, "Ridge")
    _plot_residuals(y_test, y_pred, "Ridge")
    _plot_feature_importance(model, feature_names, "Ridge")
    _save_model(model, "Ridge")
    return model, metrics


def train_lasso(X_train, X_test, y_train, y_test, feature_names) -> tuple:
    logger.info("=== Lasso Regression ===")
    model  = _tune_and_validate("Lasso", Lasso(max_iter=500000, tol=1e-3), X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = _record_metrics("Lasso", y_test, y_pred)

    _plot_actual_vs_predicted(y_test, y_pred, "Lasso")
    _plot_residuals(y_test, y_pred, "Lasso")
    _plot_feature_importance(model, feature_names, "Lasso")
    _save_model(model, "Lasso")
    return model, metrics


def train_decision_tree(X_train, X_test, y_train, y_test, feature_names) -> tuple:
    logger.info("=== Decision Tree ===")
    model  = _tune_and_validate(
        "DecisionTree", DecisionTreeRegressor(random_state=RANDOM_STATE), X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = _record_metrics("DecisionTree", y_test, y_pred)

    _plot_actual_vs_predicted(y_test, y_pred, "DecisionTree")
    _plot_residuals(y_test, y_pred, "DecisionTree")
    _plot_feature_importance(model, feature_names, "DecisionTree")
    _save_model(model, "DecisionTree")
    return model, metrics


def train_random_forest(X_train, X_test, y_train, y_test, feature_names) -> tuple:
    logger.info("=== Random Forest ===")
    model  = _tune_and_validate(
        "RandomForest", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = _record_metrics("RandomForest", y_test, y_pred)

    _plot_actual_vs_predicted(y_test, y_pred, "RandomForest")
    _plot_residuals(y_test, y_pred, "RandomForest")
    _plot_feature_importance(model, feature_names, "RandomForest")
    _save_model(model, "RandomForest")
    return model, metrics


# ── Best-model persistence & retraining ───────────────────────────────────────

def load_best_model() -> tuple:
    """Load saved best model, scaler, and metadata. Raises if run_modeling() hasn't been called."""
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("No saved model found — run run_modeling() first.")

    model  = joblib.load(model_path)
    scaler = joblib.load(MODELS_DIR / "scaler.joblib") if (MODELS_DIR / "scaler.joblib").exists() else None
    info_path = MODELS_DIR / "best_model_info.json"
    info   = json.loads(info_path.read_text()) if info_path.exists() else {}
    logger.info("Loaded best model: %s  R²=%.4f",
                info.get("name", "?"), info.get("metrics", {}).get("R²", float("nan")))
    return model, scaler, info


def retrain_best_model(df: pd.DataFrame):
    """Retrain the best model on a new/larger dataset and overwrite the saved files."""
    _, _, info = load_best_model()
    name          = info.get("name")
    feature_names = info.get("feature_names", [])
    best_params   = info.get("best_params", {})

    drop_cols = (_DROP_ALWAYS | set(df.select_dtypes(include="object").columns)) & set(df.columns)
    X = df.drop(columns=list(drop_cols))
    y = df[TARGET_COLUMN]

    if feature_names:
        missing = [c for c in feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Retrain data is missing features: {missing}")
        X = X[feature_names]

    scaler  = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    _model_factory = {
        "LinearRegression": lambda: LinearRegression(),
        "Ridge":            lambda: Ridge(max_iter=10000),
        "Lasso":            lambda: Lasso(max_iter=100000),
        "DecisionTree":     lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForest":     lambda: RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    }
    if name not in _model_factory:
        raise ValueError(f"Unknown model name in saved info: {name}")

    model = _model_factory[name]()
    if best_params:
        model.set_params(**best_params)

    model.fit(X_scaled, y)
    joblib.dump(model,  MODELS_DIR / "best_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    logger.info("Retrained %s on %d samples → models/best_model.joblib", name, len(X))
    return model


# ── Private helpers ────────────────────────────────────────────────────────────

def _prepare_data(df: pd.DataFrame) -> tuple:
    drop_cols = (_DROP_ALWAYS | set(df.select_dtypes(include="object").columns)) & set(df.columns)
    X = df.drop(columns=list(drop_cols))
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Fit scaler on train only to prevent test-set statistics leaking into training
    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns, index=X_test.index)

    logger.info("Data prepared — train: %d, test: %d, features: %d",
                len(X_train), len(X_test), X.shape[1])
    return X_train, X_test, y_train, y_test, list(X.columns), scaler


def _tune_and_validate(name: str, model, X_train, y_train):
    # KFold with shuffle ensures each fold gets a representative sample of fare ranges
    cv   = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = PARAM_GRIDS.get(name, {})

    if grid:
        # GridSearchCV refits on the full train set after finding the best params
        search = GridSearchCV(model, grid, cv=cv, scoring="r2", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        logger.info("  %s best params: %s  CV R²=%.4f", name, search.best_params_, search.best_score_)
    else:
        model.fit(X_train, y_train)
        best = model

    scores = cross_val_score(best, X_train, y_train, cv=cv, scoring="r2")
    logger.info("  %s CV R² folds: %s  mean=%.4f ± %.4f",
                name, " | ".join(f"{s:.4f}" for s in scores),
                scores.mean(), scores.std())
    return best


def _record_metrics(name: str, y_test, y_pred) -> dict:
    r2, mae, rmse = _metrics(y_test, y_pred)
    logger.info("  %s — Test  R²: %.4f  MAE: %.2f  RMSE: %.2f", name, r2, mae, rmse)
    return {"R²": round(r2, 4), "MAE": round(mae, 2), "RMSE": round(rmse, 2)}


def _metrics(y_true, y_pred) -> tuple:
    return (
        r2_score(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def _save_model(model, name: str) -> None:
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    logger.info("  Saved: models/%s.joblib", name)


def _save_best_model(model, name: str, metrics: dict, feature_names: list) -> None:
    joblib.dump(model, MODELS_DIR / "best_model.joblib")
    info = {
        "name": name,
        "metrics": metrics,
        "feature_names": feature_names,
        "best_params": {
            k: v for k, v in model.get_params().items()
            if k in {"alpha", "max_depth", "n_estimators", "min_samples_split"}
        } if hasattr(model, "get_params") else {},
    }
    (MODELS_DIR / "best_model_info.json").write_text(json.dumps(info, indent=2, default=str))
    logger.info("Best model → %s (R²=%.4f) saved to models/best_model.joblib", name, metrics["R²"])


def _comparison_table(results: dict) -> pd.DataFrame:
    table = pd.DataFrame(results).T.sort_values("R²", ascending=False)
    logger.info("=== Model Comparison ===\n%s", table.to_string())
    return table


def _plot_actual_vs_predicted(y_test, y_pred, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
    lo = min(float(np.min(y_test)), float(np.min(y_pred)))
    hi = max(float(np.max(y_test)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Fare (BDT)")
    ax.set_ylabel("Predicted Fare (BDT)")
    ax.set_title(f"{model_name} — Actual vs Predicted")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"{model_name}_actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: plots/%s_actual_vs_predicted.png", model_name)


def _plot_residuals(y_test, y_pred, model_name: str) -> None:
    residuals = np.array(y_test) - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Fare (BDT)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    pd.Series(residuals).plot.hist(bins=40, ax=axes[1], edgecolor="white", color="coral")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")

    plt.suptitle(f"{model_name} — Residual Analysis")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"{model_name}_residuals.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: plots/%s_residuals.png", model_name)


def _plot_model_comparison(comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # R²: higher is better — sort ascending so best appears at top of barh
    comparison["R²"].sort_values().plot.barh(ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title("R² (higher is better)")

    # MAE/RMSE: lower is better — sort descending so best appears at top
    comparison["MAE"].sort_values(ascending=False).plot.barh(ax=axes[1], color="coral", edgecolor="white")
    axes[1].set_title("MAE (lower is better)")

    comparison["RMSE"].sort_values(ascending=False).plot.barh(ax=axes[2], color="coral", edgecolor="white")
    axes[2].set_title("RMSE (lower is better)")

    plt.suptitle("Model Comparison")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: plots/model_comparison.png")


def _plot_feature_importance(model, feature_names: list, model_name: str) -> None:
    # Tree models expose feature_importances_; linear models expose coef_
    # Absolute value of coefficients used for linear models so direction doesn't affect ranking
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, "coef_"):
        importances = pd.Series(np.abs(model.coef_), index=feature_names)
    else:
        return

    top20 = importances.nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, len(top20) * 0.35)))
    top20.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"{model_name} — Top 20 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"{model_name}_feature_importance.png", dpi=150)
    plt.close(fig)
    logger.info("  Saved: plots/%s_feature_importance.png", model_name)


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_data
    from feature_engineering import create_features

    raw_df      = load_data("data/Flight_Price_Dataset_of_Bangladesh.csv")
    clean_df    = preprocess_data(raw_df)
    featured_df = create_features(clean_df)

    comparison = run_modeling(featured_df)
    print("\n=== Final Model Comparison ===")
    print(comparison.to_string())
