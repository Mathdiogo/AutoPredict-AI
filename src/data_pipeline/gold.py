# ============================================================
# Camada Gold - Preparação para o RAG
# ============================================================
# CONCEITO:
#   GOLD = dados prontos para consumo final (aqui: pelo RAG)
#
# Esta é a camada mais importante do ponto de vista do RAG.
# O que fazemos aqui:
#
#   1. CHUNKING: Cada linha do CSV vira um "chunk" de texto legível
#      - Em vez de um vetor enorme para o CSV inteiro, cada registro
#        vira um texto descritivo (ex: "Veículo V001: motor trocado em 50.000km")
#
#   2. EMBEDDING: Cada chunk de texto vira um vetor numérico
#      - Usamos sentence-transformers para isso
#
#   3. INDEXAÇÃO: Cada vetor é inserido no Milvus
#      - Cada dataset vai para sua própria coleção
#
# Por que chunks pequenos?
#   - O LLM tem limite de contexto (tokens)
#   - Chunks menores = busca mais precisa
#   - A pergunta compara com chunks individuais, não o arquivo todo
# ============================================================

import io
import logging
import pandas as pd
from src.database.minio_client import MinIOClient
from src.database.milvus_client import MilvusClient
from src.database.postgres_client import PostgresClient
from src.embeddings.embedder import get_embedder
from src.config import get_settings

logger = logging.getLogger(__name__)

SILVER_FILENAMES = {
    "vehicle_maintenance": "silver_vehicle_maintenance.csv",
    "car_predictive": "silver_car_predictive_maintenance.csv",
    "engine_fault": "silver_engine_fault_detection.csv",
}


# ============================================================
# FUNÇÕES DE CHUNKING
# Cada função converte uma linha do DataFrame em texto descritivo.
# O objetivo é criar texto que seja SEMANTICAMENTE RICO para busca.
# ============================================================

def _row_to_text_maintenance(row: pd.Series) -> str:
    """
    Converte uma linha do dataset de manutenção em texto descritivo.
    Colunas reais: Vehicle_Model, Mileage, Maintenance_History, Reported_Issues,
    Vehicle_Age, Fuel_Type, Tire_Condition, Brake_Condition, Battery_Status, Need_Maintenance
    """
    parts = ["[MANUTENÇÃO DE VEÍCULO]"]

    vehicle = _get_col(row, ["vehicle_id", "vehicle", "id", "car_id", "vehicle_model"])
    if vehicle:
        parts.append(f"Veículo: {vehicle}")

    date = _get_col(row, ["date", "service_date", "maintenance_date", "last_service_date"])
    if date:
        parts.append(f"Data de serviço: {date}")

    mileage = _get_col(row, ["mileage", "odometer", "km", "kilometers", "odometer_reading"])
    if mileage:
        parts.append(f"Quilometragem: {mileage}km")

    age = _get_col(row, ["vehicle_age", "age"])
    if age:
        parts.append(f"Idade do veículo: {age} anos")

    fuel = _get_col(row, ["fuel_type", "fuel", "combustivel"])
    if fuel:
        parts.append(f"Combustível: {fuel}")

    service = _get_col(row, ["service_type", "service", "maintenance_type", "type", "maintenance_history"])
    if service:
        parts.append(f"Histórico de manutenção: {service}")

    issues = _get_col(row, ["reported_issues", "issues", "problemas"])
    if issues:
        parts.append(f"Problemas reportados: {issues}")

    tire = _get_col(row, ["tire_condition", "tyre_condition"])
    if tire:
        parts.append(f"Condição dos pneus: {tire}")

    brake = _get_col(row, ["brake_condition"])
    if brake:
        parts.append(f"Condição dos freios: {brake}")

    battery = _get_col(row, ["battery_status"])
    if battery:
        parts.append(f"Status da bateria: {battery}")

    need_maint = _get_col(row, ["need_maintenance", "needs_maintenance"])
    if need_maint is not None:
        try:
            label = "NECESSITA MANUTENÇÃO" if int(float(need_maint)) == 1 else "Em dia"
            parts.append(f"Status: {label}")
        except (ValueError, TypeError):
            parts.append(f"Status manutenção: {need_maint}")

    cost = _get_col(row, ["cost", "price", "total_cost"])
    if cost and str(cost) not in ["0", "0.0", ""]:
        parts.append(f"Custo: R${cost}")

    notes = _get_col(row, ["notes", "technician_notes", "comments", "description", "observations"])
    if notes:
        parts.append(f"Observações: {notes}")

    return ". ".join(parts)


def _row_to_text_predictive(row: pd.Series) -> str:
    """
    Converte uma linha do dataset preditivo em texto descritivo.
    Colunas reais (cars_hyundai.csv, após sanitização silver):
    engine_temperature_c, brake_pad_thickness_mm, tire_pressure_psi,
    maintenance_type, anomaly_indication
    """
    parts = ["[DADOS DE SENSORES PREDITIVOS]"]

    engine_temp = _get_col(row, ["engine_temp", "engine_temperature", "motor_temp", "engine_temperature_c"])
    if engine_temp:
        try:
            temp_val = float(engine_temp)
            status = " (CRÍTICO - SUPERAQUECIMENTO)" if temp_val > 105 else " (ALTO)" if temp_val > 90 else " (NORMAL)"
            parts.append(f"Temperatura do motor: {temp_val:.1f}°C{status}")
        except (ValueError, TypeError):
            parts.append(f"Temperatura do motor: {engine_temp}")

    brake_thickness = _get_col(row, ["brake_pad_thickness", "brake_pad_thickness_mm"])
    if brake_thickness:
        try:
            val = float(brake_thickness)
            status = " (DESGASTADO - TROCA NECESSÁRIA)" if val < 3.0 else " (BOM)" if val > 6.0 else " (REGULAR)"
            parts.append(f"Espessura das pastilhas de freio: {val:.1f}mm{status}")
        except (ValueError, TypeError):
            parts.append(f"Pastilhas de freio: {brake_thickness}")

    tire_pressure = _get_col(row, ["tire_pressure", "tire_pressure_psi", "tyre_pressure"])
    if tire_pressure:
        try:
            val = float(tire_pressure)
            status = " (BAIXA)" if val < 30 else " (ALTA)" if val > 38 else " (NORMAL)"
            parts.append(f"Pressão dos pneus: {val:.1f} PSI{status}")
        except (ValueError, TypeError):
            parts.append(f"Pressão dos pneus: {tire_pressure}")

    maint_type = _get_col(row, ["maintenance_type", "service_type", "type"])
    if maint_type:
        parts.append(f"Tipo de manutenção: {maint_type}")

    anomaly = _get_col(row, ["anomaly_indication", "failure_flag", "failure", "is_failure", "fault", "falha"])
    if anomaly is not None:
        try:
            is_anomaly = int(float(str(anomaly))) == 1
            parts.append(f"Status: {'⚠️ ANOMALIA DETECTADA' if is_anomaly else '✓ NORMAL'}")
        except (ValueError, TypeError):
            parts.append(f"Status: {anomaly}")

    vehicle_id = _get_col(row, ["vehicle_id", "vehicle", "id", "car_id"])
    if vehicle_id:
        parts.append(f"Veículo: {vehicle_id}")

    return ". ".join(parts)


def _row_to_text_engine_fault(row: pd.Series) -> str:
    """
    Converte uma linha do dataset de falhas em texto descritivo.
    Colunas reais (engine_fault_detection_dataset.csv):
    Vibration_Amplitude, RMS_Vibration, Vibration_Frequency, Surface_Temperature,
    Exhaust_Temperature, Acoustic_dB, Acoustic_Frequency, Intake_Pressure,
    Exhaust_Pressure, Frequency_Band_Energy, Amplitude_Mean, Engine_Condition
    """
    parts = ["[DIAGNÓSTICO DE FALHA DE MOTOR]"]

    engine_cond = _get_col(row, ["engine_condition", "fault_code", "dtc_code", "error_code"])
    if engine_cond is not None:
        try:
            cond_val = int(float(str(engine_cond)))
            labels = {0: "NORMAL", 1: "ADVERTÊNCIA", 2: "FALHA CRÍTICA"}
            label = labels.get(cond_val, f"CONDIÇÃO {cond_val}")
            parts.append(f"Condição do motor: {label}")
        except (ValueError, TypeError):
            parts.append(f"Condição do motor: {engine_cond}")

    surface_temp = _get_col(row, ["surface_temperature", "engine_temp", "engine_temperature"])
    if surface_temp:
        try:
            val = float(surface_temp)
            status = " (CRÍTICO)" if val > 150 else " (ALTO)" if val > 100 else " (NORMAL)"
            parts.append(f"Temperatura de superfície: {val:.1f}°C{status}")
        except (ValueError, TypeError):
            parts.append(f"Temperatura de superfície: {surface_temp}")

    exhaust_temp = _get_col(row, ["exhaust_temperature", "exhaust_temp"])
    if exhaust_temp:
        try:
            val = float(exhaust_temp)
            parts.append(f"Temperatura do escapamento: {val:.1f}°C")
        except (ValueError, TypeError):
            parts.append(f"Temperatura do escapamento: {exhaust_temp}")

    vibration = _get_col(row, ["vibration_amplitude", "vibration", "vibration_level"])
    if vibration:
        try:
            val = float(vibration)
            status = " (ALTA - ATENÇÃO)" if val > 7.0 else " (NORMAL)"
            parts.append(f"Amplitude de vibração: {val:.2f}{status}")
        except (ValueError, TypeError):
            parts.append(f"Vibração: {vibration}")

    rms_vib = _get_col(row, ["rms_vibration"])
    if rms_vib:
        try:
            val = float(rms_vib)
            parts.append(f"Vibração RMS: {val:.3f}")
        except (ValueError, TypeError):
            parts.append(f"Vibração RMS: {rms_vib}")

    acoustic = _get_col(row, ["acoustic_db"])
    if acoustic:
        try:
            val = float(acoustic)
            status = " (ELEVADO)" if val > 100 else " (NORMAL)"
            parts.append(f"Nível acústico: {val:.1f} dB{status}")
        except (ValueError, TypeError):
            parts.append(f"Nível acústico: {acoustic}")

    intake_p = _get_col(row, ["intake_pressure"])
    if intake_p:
        parts.append(f"Pressão de admissão: {intake_p}")

    exhaust_p = _get_col(row, ["exhaust_pressure"])
    if exhaust_p:
        parts.append(f"Pressão do escapamento: {exhaust_p}")

    return ". ".join(parts)


def _get_col(row: pd.Series, possible_names: list[str]) -> str | None:
    """
    Tenta encontrar um valor usando múltiplos nomes possíveis de coluna.
    Retorna None se nenhum dos nomes existir ou o valor for vazio/nulo.
    """
    for name in possible_names:
        if name in row.index:
            val = row[name]
            if pd.notna(val) and str(val).strip() not in ["", "nan", "None"]:
                return str(val).strip()
    return None


# Mapeamento dataset → (função de chunking, coleção Milvus)
GOLD_CONFIG = {
    "vehicle_maintenance": {
        "silver_file": "silver_vehicle_maintenance_data.csv",
        "text_func": _row_to_text_maintenance,
        "collection": None,  # Será preenchido com settings
    },
    "car_predictive": {
        "silver_file": "silver_cars_hyundai.csv",
        "text_func": _row_to_text_predictive,
        "collection": None,
    },
    "engine_fault": {
        "silver_file": "silver_engine_fault_detection_dataset.csv",
        "text_func": _row_to_text_engine_fault,
        "collection": None,
    },
}


def process_to_gold(max_rows_per_dataset: int = 5000) -> dict[str, bool]:
    """
    Lê dados do Silver, gera chunks de texto + embeddings, indexa no Milvus.

    Args:
        max_rows_per_dataset: Limite de rows para não sobrecarregar em desenvolvimento.
                              Use None para indexar tudo.

    Returns:
        Dict com status de sucesso por dataset.
    """
    settings = get_settings()
    minio = MinIOClient()
    milvus = MilvusClient()
    postgres = PostgresClient()
    embedder = get_embedder()
    results = {}

    # Preenche as coleções com base nas configurações
    GOLD_CONFIG["vehicle_maintenance"]["collection"] = settings.milvus_collection_maintenance
    GOLD_CONFIG["car_predictive"]["collection"] = settings.milvus_collection_predictive
    GOLD_CONFIG["engine_fault"]["collection"] = settings.milvus_collection_engine

    for dataset_name, config in GOLD_CONFIG.items():
        logger.info(f"[Gold] Processando: {dataset_name}")

        silver_file = config["silver_file"]
        collection_name = config["collection"]
        text_func = config["text_func"]

        # Verifica se existe no Silver
        if not minio.object_exists(settings.minio_bucket_silver, silver_file):
            logger.warning(f"[Gold] {silver_file} não encontrado no Silver.")
            results[dataset_name] = False
            continue

        # Lê o CSV do Silver
        csv_content = minio.read_csv_as_string(settings.minio_bucket_silver, silver_file)
        if csv_content is None:
            results[dataset_name] = False
            continue

        df = pd.read_csv(io.StringIO(csv_content))

        # Limita para desenvolvimento
        if max_rows_per_dataset and len(df) > max_rows_per_dataset:
            logger.info(f"[Gold] Limitando {dataset_name} a {max_rows_per_dataset} rows (total: {len(df)})")
            df = df.head(max_rows_per_dataset)

        # PASSO 1: Gera textos descritivos para cada linha
        logger.info(f"[Gold] Gerando {len(df)} chunks de texto...")
        texts = [text_func(row) for _, row in df.iterrows()]

        # Remove chunks vazios
        valid_pairs = [(t, i) for i, t in enumerate(texts) if t and len(t) > 20]
        if not valid_pairs:
            logger.warning(f"[Gold] Nenhum chunk válido gerado para {dataset_name}")
            results[dataset_name] = False
            continue

        valid_texts = [t for t, _ in valid_pairs]
        valid_indices = [i for _, i in valid_pairs]

        # PASSO 2: Gera embeddings em batch (mais eficiente)
        embeddings = embedder.embed_batch(valid_texts)

        # PASSO 3: Prepara metadados (informações extras para exibir na resposta)
        metadata_list = []
        for i in valid_indices:
            row = df.iloc[i]
            # Pega algumas colunas-chave como metadado
            meta = {"dataset": dataset_name, "row_index": int(i)}
            for key_col in ["vehicle_id", "vehicle_model", "fault_code", "engine_condition",
                            "severity", "failure_flag", "anomaly_indication",
                            "date", "last_service_date", "service_type", "maintenance_type"]:
                if key_col in row.index and pd.notna(row[key_col]):
                    meta[key_col] = str(row[key_col])
            metadata_list.append(meta)

        # PASSO 4: Insere no Milvus
        milvus.insert(
            collection_name=collection_name,
            texts=valid_texts,
            embeddings=embeddings,
            metadata_list=metadata_list,
        )

        # Registra cada documento no PostgreSQL (log)
        for text, meta in zip(valid_texts[:10], metadata_list[:10]):  # Loga só os primeiros 10
            postgres.log_document(
                collection_name=collection_name,
                dataset_source=dataset_name,
                text_preview=text[:500],
            )

        # Salva os textos processados no MinIO Gold também
        df_gold = df.iloc[valid_indices].copy()
        df_gold["chunk_text"] = valid_texts
        minio.upload_dataframe(
            bucket=settings.minio_bucket_gold,
            object_name=f"gold_{dataset_name}.csv",
            df_csv=df_gold.to_csv(index=False),
        )

        postgres.log_ingestion(
            dataset_name=dataset_name,
            layer="gold",
            status="success",
            records_processed=len(valid_texts),
        )

        results[dataset_name] = True
        logger.info(f"[Gold] ✓ {dataset_name}: {len(valid_texts)} chunks indexados no Milvus")

    return results
