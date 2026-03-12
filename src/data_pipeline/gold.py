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
    Ex: "Manutenção do Veículo V001 em 15/01/2023 com 50000km:
         Serviço: Troca de óleo. Peças: filtro de óleo.
         Custo: R$150.00. Observações: sem anomalias."
    """
    parts = ["[MANUTENÇÃO DE VEÍCULO]"]

    # Tenta diferentes nomes de coluna (cada dataset pode ter nomes diferentes)
    vehicle_id = _get_col(row, ["vehicle_id", "vehicle", "id", "car_id"])
    if vehicle_id:
        parts.append(f"Veículo: {vehicle_id}")

    date = _get_col(row, ["date", "service_date", "maintenance_date"])
    if date:
        parts.append(f"Data: {date}")

    mileage = _get_col(row, ["mileage", "odometer", "km", "kilometers"])
    if mileage:
        parts.append(f"Quilometragem: {mileage}km")

    service = _get_col(row, ["service_type", "service", "maintenance_type", "type"])
    if service:
        parts.append(f"Serviço realizado: {service}")

    parts_replaced = _get_col(row, ["parts_replaced", "parts", "components", "replaced_parts"])
    if parts_replaced:
        parts.append(f"Peças substituídas: {parts_replaced}")

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
    Ex: "Leitura de sensores - Veículo V001:
         Temperatura do motor: 95°C (ALTA). Pressão do óleo: 2.1 bar (BAIXA).
         Vibração: 0.8g. Status: FALHA DETECTADA."
    """
    parts = ["[DADOS DE SENSORES PREDITIVOS]"]

    vehicle_id = _get_col(row, ["vehicle_id", "vehicle", "id", "car_id"])
    if vehicle_id:
        parts.append(f"Veículo: {vehicle_id}")

    timestamp = _get_col(row, ["timestamp", "datetime", "date", "time"])
    if timestamp:
        parts.append(f"Data/Hora: {timestamp}")

    # Temperatura do motor com diagnóstico automático
    engine_temp = _get_col(row, ["engine_temp", "engine_temperature", "motor_temp"])
    if engine_temp:
        try:
            temp_val = float(engine_temp)
            status = " (CRÍTICO - SUPERAQUECIMENTO)" if temp_val > 105 else " (ALTO)" if temp_val > 90 else " (NORMAL)"
            parts.append(f"Temperatura do motor: {temp_val:.1f}°C{status}")
        except (ValueError, TypeError):
            parts.append(f"Temperatura do motor: {engine_temp}")

    # Pressão do óleo
    oil_pressure = _get_col(row, ["oil_pressure", "pressao_oleo"])
    if oil_pressure:
        try:
            pressure_val = float(oil_pressure)
            status = " (BAIXA - ATENÇÃO)" if pressure_val < 2.5 else " (NORMAL)"
            parts.append(f"Pressão do óleo: {pressure_val:.2f} bar{status}")
        except (ValueError, TypeError):
            parts.append(f"Pressão do óleo: {oil_pressure}")

    vibration = _get_col(row, ["vibration", "vibration_level", "vibracao"])
    if vibration:
        parts.append(f"Vibração: {vibration}g")

    rpm = _get_col(row, ["rpm", "engine_rpm", "motor_rpm"])
    if rpm:
        parts.append(f"RPM: {rpm}")

    fuel = _get_col(row, ["fuel_consumption", "fuel", "consumo"])
    if fuel:
        parts.append(f"Consumo de combustível: {fuel}L/100km")

    # Indicador de falha
    failure = _get_col(row, ["failure_flag", "failure", "is_failure", "fault", "falha"])
    if failure is not None:
        try:
            is_failure = int(float(str(failure))) == 1
            parts.append(f"Status: {'⚠️ FALHA DETECTADA' if is_failure else '✓ NORMAL'}")
        except (ValueError, TypeError):
            parts.append(f"Status: {failure}")

    return ". ".join(parts)


def _row_to_text_engine_fault(row: pd.Series) -> str:
    """
    Converte uma linha do dataset de falhas em texto descritivo.
    Ex: "CÓDIGO DE FALHA P0300 - SEVERIDADE: ALTA:
         Descrição: Múltiplos cilindros com misfire detectado.
         RPM: 800. Temp coolant: 82°C.
         Ação recomendada: Verificar velas, bobinas e injetores."
    """
    parts = ["[DIAGNÓSTICO DE FALHA DE MOTOR]"]

    fault_code = _get_col(row, ["fault_code", "dtc_code", "error_code", "code", "fault"])
    if fault_code:
        parts.append(f"Código de falha: {fault_code}")

    severity = _get_col(row, ["severity", "fault_severity", "level", "gravidade"])
    if severity:
        parts.append(f"Severidade: {severity}")

    description = _get_col(row, ["description", "fault_description", "descricao", "fault_name"])
    if description:
        parts.append(f"Descrição: {description}")

    rpm = _get_col(row, ["engine_rpm", "rpm"])
    if rpm:
        parts.append(f"RPM do motor: {rpm}")

    coolant = _get_col(row, ["coolant_temp", "coolant_temperature", "temperatura_coolant"])
    if coolant:
        parts.append(f"Temperatura do líquido de arrefecimento: {coolant}°C")

    action = _get_col(row, ["recommended_action", "action", "recommendation", "acao_recomendada"])
    if action:
        parts.append(f"Ação recomendada: {action}")

    component = _get_col(row, ["affected_component", "component", "componente"])
    if component:
        parts.append(f"Componente afetado: {component}")

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
        "silver_file": "silver_vehicle_maintenance.csv",
        "text_func": _row_to_text_maintenance,
        "collection": None,  # Será preenchido com settings
    },
    "car_predictive": {
        "silver_file": "silver_car_predictive_maintenance.csv",
        "text_func": _row_to_text_predictive,
        "collection": None,
    },
    "engine_fault": {
        "silver_file": "silver_engine_fault_detection.csv",
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
            for key_col in ["vehicle_id", "fault_code", "severity", "failure_flag", "date", "service_type"]:
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
