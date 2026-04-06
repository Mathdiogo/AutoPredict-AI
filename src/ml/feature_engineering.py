# ============================================================
# Feature Engineering — AutoPredict AI
# ============================================================
# Prepara os 3 datasets para treinamento de modelos ML.
#
# Cada dataset tem seu próprio target e features:
#
#   Dataset 1 (vehicle_maintenance_data.csv)
#     → Target: need_maintenance (0=OK, 1=Precisa manutenção)
#     → Features: quilometragem, idade, condição dos componentes...
#
#   Dataset 2 (cars_hyundai.csv — sensores)
#     → Target: anomaly_indication (0=Normal, 1=Anomalia)
#     → Features: temperatura do motor, espessura freio, pressão pneu
#
#   Dataset 3 (engine_fault_detection_dataset.csv)
#     → Target: engine_condition (0=Normal, 1=Warning, 2=Critical)
#     → Features: vibração, temperatura, acústica, pressão
#
# Estratégia "3 datasets em conjunto":
#   Treinamos 1 modelo por dataset. Na inferência, combinamos as
#   3 probabilidades de risco em um único "Vehicle Health Index".
# ============================================================

import io
import logging
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Mapeamentos ordinais de qualidade (ordem crescente de condição) ──────────
_TIRE_MAP   = {"worn out": 0, "good": 1, "new": 2}
_BRAKE_MAP  = {"worn out": 0, "good": 1, "new": 2}
_BATTERY_MAP = {"weak": 0, "good": 1, "new": 2}
_MAINT_HIST_MAP = {"poor": 0, "average": 1, "good": 2}


# ── Utilitários de carregamento ───────────────────────────────────────────────

def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a mesma sanitização do silver.py para colunas."""
    df.columns = [
        re.sub(r"[^a-z0-9_]", "", c.strip().lower().replace(" ", "_")).strip("_")
        for c in df.columns
    ]
    return df


def _load_from_minio(silver_filename: str) -> pd.DataFrame | None:
    """Tenta carregar CSV do bucket Silver no MinIO."""
    try:
        from src.database.minio_client import MinIOClient
        from src.config import get_settings
        settings = get_settings()
        csv_str = MinIOClient().read_csv_as_string(settings.minio_bucket_silver, silver_filename)
        if csv_str:
            return pd.read_csv(io.StringIO(csv_str))
    except Exception as exc:
        logger.warning(f"MinIO indisponível para '{silver_filename}': {exc}")
    return None


def _load_from_disk(local_filename: str) -> pd.DataFrame | None:
    """Fallback: lê CSV direto de data/ e sanitiza colunas."""
    try:
        df = pd.read_csv(f"data/{local_filename}")
        return _sanitize_columns(df)
    except Exception as exc:
        logger.warning(f"Disco indisponível para '{local_filename}': {exc}")
    return None


def load_dataset(silver_filename: str, local_filename: str) -> pd.DataFrame:
    """
    Carrega dataset tentando MinIO Silver primeiro, depois disco local.
    Lança FileNotFoundError se ambos falharem.
    """
    df = _load_from_minio(silver_filename)
    if df is not None:
        logger.info(f"Carregado do MinIO Silver: {silver_filename} ({len(df)} linhas)")
        return df

    df = _load_from_disk(local_filename)
    if df is not None:
        logger.info(f"Carregado do disco (fallback): data/{local_filename} ({len(df)} linhas)")
        return df

    raise FileNotFoundError(
        f"Dataset '{silver_filename}' não encontrado no MinIO nem em data/. "
        "Execute o pipeline Bronze→Silver primeiro ou confirme os arquivos em data/."
    )


# ── Dataset 1: Vehicle Maintenance ───────────────────────────────────────────

def prepare_maintenance_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepara features do dataset de manutenção de veículos.

    Colunas reais (após silver):
        vehicle_model, mileage, maintenance_history, reported_issues,
        vehicle_age, fuel_type, transmission_type, engine_size,
        odometer_reading, last_service_date, owner_type,
        insurance_premium, service_history, accident_history,
        fuel_efficiency, tire_condition, brake_condition,
        battery_status, need_maintenance

    Target: need_maintenance (0=OK, 1=Precisa manutenção)
    """
    df = df.copy()

    target_col = "need_maintenance"
    if target_col not in df.columns:
        raise ValueError(
            f"Coluna '{target_col}' não encontrada.\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )

    y = df[target_col].astype(int)

    # ── Features numéricas diretas ────────────────────────────────────────
    num_cols = [
        "mileage", "vehicle_age", "engine_size", "odometer_reading",
        "insurance_premium", "service_history", "accident_history",
        "fuel_efficiency", "reported_issues",
    ]
    available_num = [c for c in num_cols if c in df.columns]
    X = df[available_num].copy()

    # ── Features ordinais (qualidade dos componentes) ─────────────────────
    if "tire_condition" in df.columns:
        X["tire_condition_enc"] = (
            df["tire_condition"].str.lower().map(_TIRE_MAP).fillna(1)
        )
    if "brake_condition" in df.columns:
        X["brake_condition_enc"] = (
            df["brake_condition"].str.lower().map(_BRAKE_MAP).fillna(1)
        )
    if "battery_status" in df.columns:
        X["battery_status_enc"] = (
            df["battery_status"].str.lower().map(_BATTERY_MAP).fillna(1)
        )
    if "maintenance_history" in df.columns:
        X["maintenance_history_enc"] = (
            df["maintenance_history"].str.lower().map(_MAINT_HIST_MAP).fillna(1)
        )

    # ── Features derivadas de domínio ─────────────────────────────────────
    # Score de condição geral dos componentes (0=ruim → 1=ótimo)
    enc_cols = [c for c in ["tire_condition_enc", "brake_condition_enc", "battery_status_enc"] if c in X.columns]
    if enc_cols:
        X["condition_score"] = X[enc_cols].mean(axis=1)

    # Quilômetros rodados por ano (proxy de desgaste)
    if "odometer_reading" in X.columns and "vehicle_age" in X.columns:
        X["km_per_year"] = X["odometer_reading"] / X["vehicle_age"].replace(0, 1)

    # ── One-hot encoding de categóricas nominais ──────────────────────────
    for col, prefix in [
        ("fuel_type", "fuel"),
        ("transmission_type", "trans"),
        ("owner_type", "owner"),
    ]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=prefix, drop_first=False)
            # Converte bool para int (compatibilidade scikit-learn)
            X = pd.concat([X, dummies.astype(int)], axis=1)

    X = X.fillna(X.median(numeric_only=True))
    return X, y


# ── Dataset 2: Cars Hyundai (Sensores Preditivos) ─────────────────────────────

def prepare_predictive_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepara features do dataset de sensores preditivos.

    Colunas reais (após silver, sanitizadas com regex):
        engine_temperature_c, brake_pad_thickness_mm,
        tire_pressure_psi, maintenance_type, anomaly_indication

    Target: anomaly_indication (0=Normal, 1=Anomalia)
    """
    df = df.copy()

    target_col = "anomaly_indication"
    if target_col not in df.columns:
        raise ValueError(
            f"Coluna '{target_col}' não encontrada.\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )

    y = df[target_col].astype(int)

    sensor_cols = ["engine_temperature_c", "brake_pad_thickness_mm", "tire_pressure_psi"]
    available = [c for c in sensor_cols if c in df.columns]
    X = df[available].copy().fillna(df[available].median())

    # ── Features de threshold (regras de domínio automotivo) ──────────────
    if "engine_temperature_c" in X.columns:
        X["engine_overheat"]  = (X["engine_temperature_c"] > 100).astype(int)
        X["engine_high_temp"] = (X["engine_temperature_c"] > 90).astype(int)

    if "brake_pad_thickness_mm" in X.columns:
        X["brake_worn"]     = (X["brake_pad_thickness_mm"] < 3.0).astype(int)
        X["brake_marginal"] = (X["brake_pad_thickness_mm"] < 5.0).astype(int)

    if "tire_pressure_psi" in X.columns:
        X["tire_low"]  = (X["tire_pressure_psi"] < 30.0).astype(int)
        X["tire_high"] = (X["tire_pressure_psi"] > 38.0).astype(int)
        X["tire_off"]  = (X["tire_low"] | X["tire_high"]).astype(int)

    # ── Score combinado: quantos sistemas em estado crítico ───────────────
    critical = [c for c in ["engine_overheat", "brake_worn", "tire_off"] if c in X.columns]
    if critical:
        X["critical_systems_count"] = X[critical].sum(axis=1)

    # ── One-hot do tipo de manutenção ─────────────────────────────────────
    if "maintenance_type" in df.columns:
        dummies = pd.get_dummies(df["maintenance_type"], prefix="maint", drop_first=False)
        X = pd.concat([X, dummies.astype(int)], axis=1)

    X = X.fillna(X.median(numeric_only=True))
    return X, y


# ── Dataset 3: Engine Fault Detection ────────────────────────────────────────

def prepare_engine_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepara features do dataset de falhas de motor (vibração + acústica).

    Colunas reais (após silver):
        vibration_amplitude, rms_vibration, vibration_frequency,
        surface_temperature, exhaust_temperature, acoustic_db,
        acoustic_frequency, intake_pressure, exhaust_pressure,
        frequency_band_energy, amplitude_mean, engine_condition

    Target: engine_condition (0=Normal, 1=Warning, 2=Critical)
    """
    df = df.copy()

    target_col = "engine_condition"
    if target_col not in df.columns:
        raise ValueError(
            f"Coluna '{target_col}' não encontrada.\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )

    y = df[target_col].astype(int)

    raw_cols = [
        "vibration_amplitude", "rms_vibration", "vibration_frequency",
        "surface_temperature", "exhaust_temperature",
        "acoustic_db", "acoustic_frequency",
        "intake_pressure", "exhaust_pressure",
        "frequency_band_energy", "amplitude_mean",
    ]
    available = [c for c in raw_cols if c in df.columns]
    X = df[available].copy().fillna(df[available].median())

    # ── Features derivadas de física do motor ────────────────────────────
    if "vibration_amplitude" in X.columns and "rms_vibration" in X.columns:
        # Razão pico/RMS: alta = impacto isolado; baixa = vibração contínua
        X["vibration_crest_factor"] = X["vibration_amplitude"] / (X["rms_vibration"] + 1e-9)
        X["vibration_energy"] = X["vibration_amplitude"] ** 2

    if "surface_temperature" in X.columns and "exhaust_temperature" in X.columns:
        # Gradiente térmico: diferença grande pode indicar vazamento/bloqueio
        X["thermal_delta"] = X["exhaust_temperature"] - X["surface_temperature"]

    if "intake_pressure" in X.columns and "exhaust_pressure" in X.columns:
        # Razão de pressão: indica eficiência da combustão
        X["pressure_ratio"] = X["exhaust_pressure"] / (X["intake_pressure"] + 1e-9)

    if "acoustic_db" in X.columns:
        X["acoustic_high"] = (X["acoustic_db"] > 100).astype(int)

    if "vibration_amplitude" in X.columns:
        X["vib_critical"] = (X["vibration_amplitude"] > 7.0).astype(int)

    X = X.fillna(X.median(numeric_only=True))
    return X, y


# ── Split + Escalonamento ─────────────────────────────────────────────────────

def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]:
    """
    Divide treino/teste e normaliza com StandardScaler.

    Returns:
        (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train.values, y_test.values, feature_names, scaler


# ── Carrega os 3 datasets de uma vez ─────────────────────────────────────────

def get_all_datasets() -> dict:
    """
    Carrega e prepara os 3 datasets para treinamento.

    Returns:
        {
            "maintenance": {X_train, X_test, y_train, y_test,
                            feature_names, scaler, target_names, task, description},
            "predictive":  {...},
            "engine":      {...},
        }
    """
    datasets = {}

    # ── Dataset 1: Vehicle Maintenance ────────────────────────────────────
    try:
        df = load_dataset("silver_vehicle_maintenance_data.csv", "vehicle_maintenance_data.csv")
        X, y = prepare_maintenance_features(df)
        X_tr, X_te, y_tr, y_te, feats, scaler = split_and_scale(X, y)
        datasets["maintenance"] = {
            "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te,
            "feature_names": feats, "scaler": scaler,
            "target_names": ["OK", "Needs Maintenance"],
            "task": "binary_classification",
            "description": "Previsão de necessidade de manutenção do veículo",
        }
        logger.info(
            f"[FE] maintenance → {X_tr.shape[0]} treino | "
            f"{X_te.shape[0]} teste | {len(feats)} features"
        )
    except Exception as exc:
        logger.error(f"[FE] Falha em 'maintenance': {exc}")

    # ── Dataset 2: Sensores Preditivos ────────────────────────────────────
    try:
        df = load_dataset("silver_cars_hyundai.csv", "cars_hyundai.csv")
        X, y = prepare_predictive_features(df)
        X_tr, X_te, y_tr, y_te, feats, scaler = split_and_scale(X, y)
        datasets["predictive"] = {
            "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te,
            "feature_names": feats, "scaler": scaler,
            "target_names": ["Normal", "Anomaly"],
            "task": "binary_classification",
            "description": "Detecção de anomalia por sensores (temperatura, freio, pressão)",
        }
        logger.info(
            f"[FE] predictive → {X_tr.shape[0]} treino | "
            f"{X_te.shape[0]} teste | {len(feats)} features"
        )
    except Exception as exc:
        logger.error(f"[FE] Falha em 'predictive': {exc}")

    # ── Dataset 3: Engine Fault Detection ────────────────────────────────
    try:
        df = load_dataset("silver_engine_fault_detection_dataset.csv", "engine_fault_detection_dataset.csv")
        X, y = prepare_engine_features(df)
        X_tr, X_te, y_tr, y_te, feats, scaler = split_and_scale(X, y)
        datasets["engine"] = {
            "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te,
            "feature_names": feats, "scaler": scaler,
            "target_names": ["Normal", "Warning", "Critical"],
            "task": "multiclass_classification",
            "description": "Diagnóstico de condição do motor por vibração e temperatura",
        }
        logger.info(
            f"[FE] engine → {X_tr.shape[0]} treino | "
            f"{X_te.shape[0]} teste | {len(feats)} features"
        )
    except Exception as exc:
        logger.error(f"[FE] Falha em 'engine': {exc}")

    return datasets
