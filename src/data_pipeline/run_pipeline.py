# ============================================================
# Pipeline Orchestrator - Executa Bronze → Silver → Gold
# ============================================================
# Este script é o ponto de entrada para rodar o pipeline completo.
# Execute com: docker exec autopredict-api python /app/src/data_pipeline/run_pipeline.py
# ============================================================

import logging
import sys
import time

# Configura logging para mostrar no terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def run_full_pipeline(data_dir: str = "/app/data", max_rows: int = 5000):
    """
    Executa o pipeline completo: Bronze → Silver → Gold

    Args:
        data_dir: Diretório onde estão os CSVs baixados do Kaggle
        max_rows: Limite de linhas por dataset (None = sem limite)
    """
    from src.data_pipeline.bronze import ingest_to_bronze
    from src.data_pipeline.silver import process_to_silver
    from src.data_pipeline.gold import process_to_gold

    logger.info("=" * 60)
    logger.info("  AutoPredict AI - Pipeline de Ingestão")
    logger.info("=" * 60)

    # ── BRONZE ──────────────────────────────────────────────
    logger.info("\n[1/3] CAMADA BRONZE - Ingestão dos dados brutos")
    start = time.time()
    bronze_results = ingest_to_bronze(data_dir=data_dir)
    logger.info(f"Bronze concluído em {time.time() - start:.1f}s: {bronze_results}")

    if not any(bronze_results.values()):
        logger.error(
            "\n❌ Nenhum dataset encontrado!\n"
            f"Por favor, baixe os datasets do Kaggle e coloque em: {data_dir}/\n\n"
            "Arquivos esperados:\n"
            "  - vehicle_maintenance.csv\n"
            "  - car_predictive_maintenance.csv\n"
            "  - engine_fault_detection.csv\n\n"
            "Veja data/README.md para instruções detalhadas."
        )
        sys.exit(1)

    # ── SILVER ──────────────────────────────────────────────
    logger.info("\n[2/3] CAMADA SILVER - Limpeza e padronização")
    start = time.time()
    silver_results = process_to_silver()
    logger.info(f"Silver concluído em {time.time() - start:.1f}s: {silver_results}")

    # ── GOLD ────────────────────────────────────────────────
    logger.info("\n[3/3] CAMADA GOLD - Embeddings + Indexação no Milvus")
    start = time.time()
    gold_results = process_to_gold(max_rows_per_dataset=max_rows)
    logger.info(f"Gold concluído em {time.time() - start:.1f}s: {gold_results}")

    # ── RESUMO ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  RESUMO DO PIPELINE")
    logger.info("=" * 60)
    for dataset in ["vehicle_maintenance", "car_predictive", "engine_fault"]:
        b = "✓" if bronze_results.get(dataset) else "✗"
        s = "✓" if silver_results.get(dataset) else "✗"
        g = "✓" if gold_results.get(dataset) else "✗"
        logger.info(f"  {dataset:<30} Bronze:{b}  Silver:{s}  Gold:{g}")

    success_count = sum(1 for v in gold_results.values() if v)
    logger.info(f"\n  {success_count}/3 datasets indexados com sucesso!")
    logger.info("  O sistema está pronto para receber perguntas.\n")


if __name__ == "__main__":
    # Suporte a argumentos: python run_pipeline.py --data-dir /caminho --max-rows 1000
    import argparse

    parser = argparse.ArgumentParser(description="AutoPredict AI - Pipeline de Ingestão")
    parser.add_argument("--data-dir", default="/app/data", help="Diretório com os CSVs")
    parser.add_argument("--max-rows", type=int, default=5000, help="Máximo de linhas por dataset (0 = sem limite)")
    args = parser.parse_args()

    max_rows = args.max_rows if args.max_rows > 0 else None
    run_full_pipeline(data_dir=args.data_dir, max_rows=max_rows)
