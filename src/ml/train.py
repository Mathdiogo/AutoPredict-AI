# ============================================================
# Pipeline de Treinamento ML — AutoPredict AI  (Sprint 4 / AC1)
# ============================================================
# Treina 3 modelos para cada dataset e registra tudo no MLflow.
#
# Modelos treinados:
#   1. Logistic Regression (baseline linear)
#   2. Random Forest       (ensemble de árvores)
#   3. XGBoost             (gradient boosting — geralmente o mais forte)
#
# Para cada dataset:
#   - Cria um Experiment no MLflow
#   - Registra parâmetros, métricas e artefatos de cada run
#   - Registra o melhor modelo no Model Registry
#
# Experimentos:
#   AutoPredict-Maintenance  → need_maintenance (binário)
#   AutoPredict-Predictive   → anomaly_indication (binário)
#   AutoPredict-EngineFault  → engine_condition (multiclasse 0/1/2)
#
# Para rodar:
#   docker exec autopredict-api python /app/src/ml/train.py
# ============================================================

import logging
import os

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.ml.feature_engineering import get_all_datasets

logger = logging.getLogger(__name__)

# ── Nomes dos experimentos no MLflow ─────────────────────────────────────────
EXPERIMENT_NAMES = {
    "maintenance": "AutoPredict-Maintenance",
    "predictive":  "AutoPredict-Predictive",
    "engine":      "AutoPredict-EngineFault",
}

# ── Configuração dos modelos ─────────────────────────────────────────────────
# Cada modelo é uma factory function que recebe n_classes para configurar
# corretamente classificação binária vs multiclasse.

def _build_models(n_classes: int) -> dict:
    """Retorna dicionário de modelos configurados para n_classes."""
    is_multi = n_classes > 2
    return {
        "logistic_regression": {
            "model": LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="multinomial" if is_multi else "auto",
            ),
            "params": {
                "max_iter": 1000,
                "class_weight": "balanced",
                "solver": "lbfgs",
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
            },
        },
        "xgboost": {
            "model": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss" if is_multi else "logloss",
                # num_class só para multiclasse; None quebra o XGBoost
                **({"num_class": n_classes} if is_multi else {}),
                n_jobs=-1,
                verbosity=0,
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
            },
        },
    }


# ── Métricas ─────────────────────────────────────────────────────────────────

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
) -> dict:
    """Calcula accuracy, precision, recall, F1 e ROC-AUC."""
    avg = "macro" if n_classes > 2 else "binary"

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall":    recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1_score":  f1_score(y_true, y_pred, average=avg, zero_division=0),
    }

    try:
        if n_classes == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
    except Exception:
        metrics["roc_auc"] = 0.0

    return metrics


# ── Treino de um único modelo ─────────────────────────────────────────────────

def _train_single_model(
    model_name: str,
    model_cfg: dict,
    data: dict,
    n_classes: int,
    experiment_name: str,
) -> dict:
    """
    Treina um modelo, registra no MLflow e retorna métricas.
    """
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    feature_names = data["feature_names"]

    with mlflow.start_run(run_name=model_name):
        # ── Parâmetros ────────────────────────────────────────────────────
        mlflow.log_param("dataset",          data.get("description", ""))
        mlflow.log_param("model",            model_name)
        mlflow.log_param("n_features",       len(feature_names))
        mlflow.log_param("n_train_samples",  len(X_train))
        mlflow.log_param("n_test_samples",   len(X_test))
        mlflow.log_param("n_classes",        n_classes)
        for k, v in model_cfg["params"].items():
            mlflow.log_param(k, v)

        # ── Treino ────────────────────────────────────────────────────────
        model = model_cfg["model"]
        # Para XGBoost multiclasse: passa sample_weight (não tem class_weight nativo).
        # RF e LR já têm class_weight="balanced" — não duplicar o balanceamento.
        fit_kwargs = {}
        if n_classes > 2 and getattr(model, "class_weight", None) is None:
            from sklearn.utils.class_weight import compute_sample_weight
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, **fit_kwargs)

        # ── Predição ──────────────────────────────────────────────────────
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # ── Métricas ──────────────────────────────────────────────────────
        metrics = _compute_metrics(y_test, y_pred, y_proba, n_classes)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # ── Importância de features (Random Forest / XGBoost) ────────────
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(feature_names, model.feature_importances_))
            top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
            for feat, imp in top:
                # Trunca nome para caber no MLflow
                mlflow.log_metric(f"feat_imp__{feat[:40]}", round(float(imp), 6))

        # ── Salva modelo no MLflow ────────────────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"{experiment_name}__{model_name}",
        )

        # ── Log do relatório de classificação como artefato texto ─────────
        report = classification_report(
            y_test, y_pred,
            target_names=data.get("target_names"),
            zero_division=0,
        )
        mlflow.log_text(report, "classification_report.txt")

        run_id = mlflow.active_run().info.run_id

    logger.info(
        f"    {model_name:22s} | "
        f"Acc={metrics['accuracy']:.4f} | "
        f"F1={metrics['f1_score']:.4f} | "
        f"AUC={metrics['roc_auc']:.4f}"
    )

    return {"model_name": model_name, "run_id": run_id, "metrics": metrics, "model": model}


# ── Orquestrador principal ────────────────────────────────────────────────────

def train_all_models(mlflow_tracking_uri: str = "http://localhost:5001") -> dict:
    """
    Treina os 3 modelos para cada dataset e registra no MLflow.

    Returns:
        {
            "maintenance": {best_model, best_run_id, best_metrics, experiment},
            "predictive":  {...},
            "engine":      {...},
        }
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")

    logger.info("Carregando e preparando os 3 datasets...")
    all_data = get_all_datasets()

    if not all_data:
        raise RuntimeError(
            "Nenhum dataset foi carregado. "
            "Execute o pipeline Bronze→Silver primeiro (S3-10)."
        )

    results = {}

    for dataset_name, data in all_data.items():
        experiment_name = EXPERIMENT_NAMES.get(dataset_name, f"AutoPredict-{dataset_name}")

        logger.info(f"\n{'='*60}")
        logger.info(f"  Experimento: {experiment_name}")
        logger.info(f"  Tarefa: {data['task']}")
        logger.info(f"  Descrição: {data['description']}")
        logger.info(f"{'='*60}")

        mlflow.set_experiment(experiment_name)

        n_classes = len(np.unique(np.concatenate([data["y_train"], data["y_test"]])))
        models = _build_models(n_classes)

        best_run = None
        best_f1  = -1.0

        for model_name, model_cfg in models.items():
            run = _train_single_model(
                model_name, model_cfg, data, n_classes, experiment_name
            )
            if run["metrics"]["f1_score"] > best_f1:
                best_f1  = run["metrics"]["f1_score"]
                best_run = run

        results[dataset_name] = {
            "best_model":   best_run["model_name"],
            "best_run_id":  best_run["run_id"],
            "best_metrics": best_run["metrics"],
            "experiment":   experiment_name,
        }

        logger.info(
            f"\n  ★ Melhor: {best_run['model_name']} "
            f"(F1={best_f1:.4f}, AUC={best_run['metrics']['roc_auc']:.4f})"
        )

    # ── Resumo final ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  RESUMO FINAL DOS EXPERIMENTOS")
    logger.info("=" * 60)
    for ds, res in results.items():
        logger.info(
            f"  {ds:14s} | melhor={res['best_model']:22s} | "
            f"F1={res['best_metrics']['f1_score']:.4f} | "
            f"AUC={res['best_metrics']['roc_auc']:.4f}"
        )

    return results


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    results = train_all_models(mlflow_tracking_uri=tracking_uri)

    print("\n✓ Treinamento concluído!")
    for ds, res in results.items():
        print(
            f"  {ds}: {res['best_model']} | "
            f"F1={res['best_metrics']['f1_score']:.4f} | "
            f"AUC={res['best_metrics']['roc_auc']:.4f}"
        )
    print(f"\nAcesse o MLflow em: http://localhost:5001")
