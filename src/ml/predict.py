# ============================================================
# Vehicle Health Index — Inferência Combinada dos 3 Modelos
# ============================================================
# Esta é a parte mais interessante do projeto:
# usamos os 3 modelos treinados em CONJUNTO para calcular
# um índice unificado de saúde do veículo.
#
# CONCEITO:
#   Cada modelo olha para um aspecto diferente do veículo:
#
#   ┌─────────────────────────────────────────────────────┐
#   │  Modelo 1 (Maintenance)                             │
#   │  "Com base no histórico, precisa de manutenção?"    │
#   │  → risco_manutencao = 0.0 a 1.0                    │
#   ├─────────────────────────────────────────────────────┤
#   │  Modelo 2 (Predictive/Sensores)                     │
#   │  "Os sensores indicam alguma anomalia?"             │
#   │  → risco_sensores = 0.0 a 1.0                      │
#   ├─────────────────────────────────────────────────────┤
#   │  Modelo 3 (Engine Fault)                            │
#   │  "Há sinais de falha no motor (vibração/acústica)?" │
#   │  → risco_motor = 0.0 a 1.0                         │
#   └─────────────────────────────────────────────────────┘
#           │               │               │
#           └───────────────┴───────────────┘
#                           │
#                   vehicle_health_index
#                   (0.0=perfeito → 1.0=crítico)
#
# Exemplo de uso:
#   predict = VehicleHealthPredictor()
#   result = predict.assess({
#       "maintenance": {...dados de manutenção...},
#       "sensors":     {...dados de sensores...},
#       "engine":      {...dados de vibração...},
#   })
#   print(result.summary())
# ============================================================

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Pesos de cada modelo no índice combinado.
# Motor tem peso maior por ser o componente mais crítico.
_RISK_WEIGHTS = {
    "maintenance": 0.25,
    "predictive":  0.35,
    "engine":      0.40,
}

# Limiares do índice de risco combinado
_RISK_THRESHOLDS = {
    "low":      (0.00, 0.30),
    "moderate": (0.30, 0.60),
    "high":     (0.60, 0.80),
    "critical": (0.80, 1.00),
}


@dataclass
class ModelRiskResult:
    """Resultado de risco de um único modelo."""
    dataset:       str
    risk_score:    float          # 0.0 (sem risco) → 1.0 (risco máximo)
    raw_proba:     list[float]    # probabilidades brutas por classe
    prediction:    int            # classe predita
    available:     bool = True    # False se modelo não carregado


@dataclass
class VehicleHealthResult:
    """Resultado consolidado da avaliação de saúde do veículo."""
    vehicle_health_index: float                           # 0.0 = perfeito, 1.0 = crítico
    risk_level:           str                             # low / moderate / high / critical
    model_results:        dict[str, ModelRiskResult] = field(default_factory=dict)
    active_models:        int = 0

    def summary(self) -> str:
        """Retorna um resumo legível da avaliação."""
        lines = [
            "=" * 50,
            f"  VEHICLE HEALTH INDEX: {self.vehicle_health_index:.2%}",
            f"  Nível de risco:        {self.risk_level.upper()}",
            f"  Modelos ativos:        {self.active_models}/3",
            "-" * 50,
        ]
        for ds, res in self.model_results.items():
            if res.available:
                lines.append(
                    f"  {ds:12s} → risco={res.risk_score:.2%}  "
                    f"(classe={res.prediction}, proba={[f'{p:.2f}' for p in res.raw_proba]})"
                )
            else:
                lines.append(f"  {ds:12s} → modelo não disponível")
        lines.append("=" * 50)
        return "\n".join(lines)


class VehicleHealthPredictor:
    """
    Carrega os melhores modelos treinados de cada experimento e
    calcula o Vehicle Health Index combinado.
    """

    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5001"):
        self.tracking_uri = mlflow_tracking_uri
        self._models: dict = {}
        self._load_models()

    def _load_models(self) -> None:
        """Carrega o melhor modelo de cada experimento do MLflow Model Registry."""
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)

        registry_names = {
            "maintenance": None,
            "predictive":  None,
            "engine":      None,
        }

        # Tenta descobrir o melhor modelo de cada experimento
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        experiment_map = {
            "maintenance": "AutoPredict-Maintenance",
            "predictive":  "AutoPredict-Predictive",
            "engine":      "AutoPredict-EngineFault",
        }

        for dataset_key, experiment_name in experiment_map.items():
            try:
                experiment = client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    logger.warning(f"Experimento '{experiment_name}' não encontrado no MLflow.")
                    continue

                # Busca o run com maior F1
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.f1_score DESC"],
                    max_results=1,
                )
                if not runs:
                    continue

                best_run = runs[0]
                model_uri = f"runs:/{best_run.info.run_id}/model"
                self._models[dataset_key] = mlflow.sklearn.load_model(model_uri)
                logger.info(
                    f"Modelo '{dataset_key}' carregado do run {best_run.info.run_id} "
                    f"(F1={best_run.data.metrics.get('f1_score', 0):.4f})"
                )
            except Exception as exc:
                logger.warning(f"Não foi possível carregar modelo '{dataset_key}': {exc}")

    # ── Funções de preparação de entrada ──────────────────────────────────

    def _prepare_single(self, raw_data: dict, dataset_key: str) -> pd.DataFrame | None:
        """
        Converte um dict de features brutas para DataFrame pronto para predição.
        Usa as mesmas funções de feature engineering do treino.
        """
        try:
            from src.ml.feature_engineering import (
                prepare_maintenance_features,
                prepare_predictive_features,
                prepare_engine_features,
                _sanitize_columns,
            )

            df = pd.DataFrame([raw_data])
            df = _sanitize_columns(df)

            # Adiciona coluna target dummy para não quebrar as funções de FE
            target_map = {
                "maintenance": "need_maintenance",
                "predictive":  "anomaly_indication",
                "engine":      "engine_condition",
            }
            target_col = target_map[dataset_key]
            if target_col not in df.columns:
                df[target_col] = 0  # dummy — não usado na predição

            fe_fn = {
                "maintenance": prepare_maintenance_features,
                "predictive":  prepare_predictive_features,
                "engine":      prepare_engine_features,
            }[dataset_key]

            X, _ = fe_fn(df)
            return X

        except Exception as exc:
            logger.error(f"Erro ao preparar features '{dataset_key}': {exc}")
            return None

    # ── Avaliação de um único modelo ──────────────────────────────────────

    def _predict_single(self, raw_data: dict, dataset_key: str) -> ModelRiskResult:
        """Executa o modelo de um dataset e retorna o resultado de risco."""
        model = self._models.get(dataset_key)
        if model is None:
            return ModelRiskResult(
                dataset=dataset_key, risk_score=0.0,
                raw_proba=[], prediction=-1, available=False,
            )

        X = self._prepare_single(raw_data, dataset_key)
        if X is None:
            return ModelRiskResult(
                dataset=dataset_key, risk_score=0.0,
                raw_proba=[], prediction=-1, available=False,
            )

        proba = model.predict_proba(X)[0]
        pred  = int(model.predict(X)[0])

        # Risco = probabilidade de classe não-normal
        # Para binário: prob(classe=1)
        # Para multiclasse: 1 - prob(classe=0) = prob(warning ou critical)
        if len(proba) == 2:
            risk_score = float(proba[1])
        else:
            risk_score = float(1.0 - proba[0])  # prob de não ser "Normal"

        return ModelRiskResult(
            dataset=dataset_key,
            risk_score=risk_score,
            raw_proba=[round(float(p), 4) for p in proba],
            prediction=pred,
            available=True,
        )

    # ── Avaliação combinada ───────────────────────────────────────────────

    def assess(self, sensor_inputs: dict) -> VehicleHealthResult:
        """
        Avalia o estado de saúde do veículo combinando os 3 modelos.

        Args:
            sensor_inputs: dict com chaves "maintenance", "sensors", "engine"
                           cada uma contendo os dados brutos do veículo.

        Returns:
            VehicleHealthResult com o índice e detalhes por modelo.

        Exemplo:
            result = predictor.assess({
                "maintenance": {
                    "Vehicle_Model": "Truck", "Mileage": 80000,
                    "Maintenance History": "Poor", ...
                },
                "sensors": {
                    "Engine Temperature (°C)": 105,
                    "Brake Pad Thickness (mm)": 2.5,
                    "Tire Pressure (PSI)": 28,
                },
                "engine": {
                    "Vibration_Amplitude": 8.5,
                    "RMS_Vibration": 3.1,
                    ...
                },
            })
        """
        # Mapa entre chaves de entrada e chaves internas
        input_key_map = {
            "maintenance": "maintenance",
            "sensors":     "predictive",
            "engine":      "engine",
        }

        model_results = {}
        weighted_risks = []
        total_weight   = 0.0

        for input_key, dataset_key in input_key_map.items():
            raw = sensor_inputs.get(input_key, {})
            if not raw:
                model_results[dataset_key] = ModelRiskResult(
                    dataset=dataset_key, risk_score=0.0,
                    raw_proba=[], prediction=-1, available=False,
                )
                continue

            result = self._predict_single(raw, dataset_key)
            model_results[dataset_key] = result

            if result.available:
                w = _RISK_WEIGHTS[dataset_key]
                weighted_risks.append(result.risk_score * w)
                total_weight += w

        # Normaliza pelo peso total dos modelos disponíveis
        if total_weight > 0:
            vhi = sum(weighted_risks) / total_weight
        else:
            vhi = 0.0

        vhi = float(np.clip(vhi, 0.0, 1.0))

        # Classifica nível de risco
        risk_level = "low"
        for level, (low, high) in _RISK_THRESHOLDS.items():
            if low <= vhi < high:
                risk_level = level
                break
        if vhi >= 1.0:
            risk_level = "critical"

        active = sum(1 for r in model_results.values() if r.available)

        return VehicleHealthResult(
            vehicle_health_index=round(vhi, 4),
            risk_level=risk_level,
            model_results=model_results,
            active_models=active,
        )
