# ============================================================
# Camada Silver - Limpeza e Padronização dos Dados
# ============================================================
# CONCEITO:
#   SILVER = dados limpos, confiáveis, com schema conhecido
#
# O que fazemos aqui:
#   - Remove duplicatas
#   - Trata valores nulos (remove ou preenche com padrão)
#   - Padroniza nomes de colunas (snake_case, sem espaços)
#   - Garante tipos corretos (datas como datetime, números como float)
#   - Filtra outliers óbvios
#
# Cada dataset tem sua função de limpeza específica,
# pois cada um tem colunas e problemas diferentes.
# ============================================================

import io
import logging
import re
import pandas as pd
from src.database.minio_client import MinIOClient
from src.database.postgres_client import PostgresClient
from src.config import get_settings

logger = logging.getLogger(__name__)


def _clean_vehicle_maintenance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset 1: Vehicle Maintenance Data
    Colunas típicas: vehicle_id, date, mileage, service_type, parts_replaced, cost
    """
    # Padroniza nomes de colunas (remove espaços, lowercase)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Remove linhas completamente vazias
    df = df.dropna(how="all")

    # Garante que colunas obrigatórias existam (trata KeyError graciosamente)
    required_cols = {"vehicle_model", "need_maintenance"}
    existing = set(df.columns)
    missing = required_cols - existing
    if missing:
        logger.warning(f"Colunas ausentes em vehicle_maintenance: {missing}")

    # Remove duplicatas
    df = df.drop_duplicates()

    # Preenche notas nulas com string vazia
    for col in ["notes", "technician_notes", "comments", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Garante que custo seja numérico (remove símbolos como $)
    if "cost" in df.columns:
        df["cost"] = pd.to_numeric(
            df["cost"].astype(str).str.replace(r"[^\d.]", "", regex=True),
            errors="coerce",
        ).fillna(0.0)

    # Garante que quilometragem seja numérica
    if "mileage" in df.columns:
        df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce").fillna(0)

    logger.info(f"[Silver] vehicle_maintenance: {len(df)} registros após limpeza")
    return df


def _clean_car_predictive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset 2: Car Predictive Maintenance Data
    Colunas reais (cars_hyundai.csv): Engine Temperature (°C),
    Brake Pad Thickness (mm), Tire Pressure (PSI), Maintenance Type, Anomaly Indication
    """
    # Sanitização extendida: remove caracteres especiais (°, (), %) além de espaços
    df.columns = [
        re.sub(r'[^a-z0-9_]', '', c.strip().lower().replace(' ', '_')).strip('_')
        for c in df.columns
    ]
    df = df.dropna(how="all").drop_duplicates()

    # Converte colunas numéricas de sensores (nomes após sanitização)
    sensor_cols = [
        "engine_temp", "engine_temperature", "engine_temperature_c",
        "oil_pressure",
        "vibration",
        "speed", "vehicle_speed",
        "fuel_consumption",
        "rpm", "engine_rpm",
        "coolant_temp", "coolant_temperature",
        "brake_pad_thickness_mm",
        "tire_pressure_psi",
    ]
    for col in sensor_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove leituras impossíveis (ex: temperatura negativa)
    if "engine_temp" in df.columns:
        df = df[(df["engine_temp"].isna()) | (df["engine_temp"].between(-50, 300))]
    if "engine_temperature" in df.columns:
        df = df[(df["engine_temperature"].isna()) | (df["engine_temperature"].between(-50, 300))]
    if "engine_temperature_c" in df.columns:
        df = df[(df["engine_temperature_c"].isna()) | (df["engine_temperature_c"].between(-50, 300))]

    # Garante que failure_flag/anomaly_indication seja binário (0 ou 1)
    for flag_col in ["failure_flag", "failure", "is_failure", "fault", "anomaly_indication"]:
        if flag_col in df.columns:
            df[flag_col] = df[flag_col].fillna(0).astype(int).clip(0, 1)
            break

    # Preenche nulos de sensores com a mediana (evita distorcer a média)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    logger.info(f"[Silver] car_predictive: {len(df)} registros após limpeza")
    return df


def _clean_engine_fault(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset 3: Engine Fault Detection Data
    Colunas típicas: fault_code, description, engine_rpm, coolant_temp,
                     severity, recommended_action
    """
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(how="all").drop_duplicates()

    # Padroniza severidade (garante que seja texto uppercase)
    for sev_col in ["severity", "fault_severity", "level"]:
        if sev_col in df.columns:
            df[sev_col] = df[sev_col].fillna("UNKNOWN").astype(str).str.upper().str.strip()
            break

    # Padroniza campos de texto
    for text_col in ["description", "fault_description", "recommended_action", "action"]:
        if text_col in df.columns:
            df[text_col] = df[text_col].fillna("").astype(str).str.strip()

    # Padroniza códigos de falha (ex: P0300 → sem espaços extras)
    for code_col in ["fault_code", "dtc_code", "error_code", "code"]:
        if code_col in df.columns:
            df[code_col] = df[code_col].astype(str).str.upper().str.strip()
            break

    logger.info(f"[Silver] engine_fault: {len(df)} registros após limpeza")
    return df


# Mapeamento: dataset → função de limpeza
CLEANERS = {
    "vehicle_maintenance": _clean_vehicle_maintenance,
    "car_predictive": _clean_car_predictive,
    "engine_fault": _clean_engine_fault,
}

BRONZE_FILENAMES = {
    "vehicle_maintenance": "vehicle_maintenance_data.csv",
    "car_predictive": "cars_hyundai.csv",
    "engine_fault": "engine_fault_detection_dataset.csv",
}


def process_to_silver() -> dict[str, bool]:
    """
    Lê os dados brutos do Bronze, limpa e salva no Silver.
    Retorna dict com status de sucesso por dataset.
    """
    settings = get_settings()
    minio = MinIOClient()
    postgres = PostgresClient()
    results = {}

    for dataset_name, filename in BRONZE_FILENAMES.items():
        logger.info(f"[Silver] Processando: {dataset_name}")

        # Verifica se o arquivo existe no Bronze
        if not minio.object_exists(settings.minio_bucket_bronze, filename):
            logger.warning(f"[Silver] {filename} não encontrado no Bronze. Execute o Bronze primeiro.")
            results[dataset_name] = False
            continue

        # Lê o CSV do MinIO Bronze
        csv_content = minio.read_csv_as_string(settings.minio_bucket_bronze, filename)
        if csv_content is None:
            results[dataset_name] = False
            continue

        # Carrega como DataFrame
        df = pd.read_csv(io.StringIO(csv_content))
        logger.info(f"[Silver] {dataset_name}: {len(df)} linhas brutas")

        # Aplica a limpeza específica do dataset
        clean_func = CLEANERS.get(dataset_name)
        if clean_func:
            df = clean_func(df)

        # Salva no MinIO Silver
        cleaned_csv = df.to_csv(index=False)
        silver_filename = f"silver_{filename}"
        success = minio.upload_dataframe(
            bucket=settings.minio_bucket_silver,
            object_name=silver_filename,
            df_csv=cleaned_csv,
        )

        postgres.log_ingestion(
            dataset_name=dataset_name,
            layer="silver",
            status="success" if success else "error",
            records_processed=len(df) if success else 0,
        )

        results[dataset_name] = success
        if success:
            logger.info(f"[Silver] ✓ {dataset_name}: {len(df)} registros limpos")

    return results
