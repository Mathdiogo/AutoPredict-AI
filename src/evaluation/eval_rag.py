"""
Avaliação quantitativa do pipeline RAG.

Métricas coletadas:
  - Retrieval: scores de similaridade por coleção, cobertura multi-source
  - Generation: tempo de resposta, tamanho da resposta
  - Qualidade: hit-rate por dataset, consistência multi-run

Uso:
  docker exec -w /app autopredict-api python -m src.evaluation.eval_rag
"""

import json
import logging
import statistics
import time

logging.basicConfig(level=logging.WARNING)  # silencia logs internos durante avaliação

# ── Perguntas de avaliação ─────────────────────────────────────────────────────
# Cobrindo os 3 domínios: manutenção, sensores, falhas de motor

EVAL_QUERIES = [
    # Domínio 1 – Manutenção histórica
    {"query": "Quais sao os problemas mais comuns em veiculos com alta quilometragem?",       "domain": "maintenance"},
    {"query": "Qual a frequencia ideal de troca de oleo do motor?",                           "domain": "maintenance"},
    {"query": "Como saber se o freio precisa de manutencao?",                                 "domain": "maintenance"},
    # Domínio 2 – Sensores preditivos
    {"query": "Qual temperatura do motor indica superaquecimento?",                           "domain": "predictive"},
    {"query": "Qual a pressao ideal dos pneus?",                                              "domain": "predictive"},
    {"query": "Espessura minima de pastilha de freio antes de trocar",                        "domain": "predictive"},
    # Domínio 3 – Falhas de motor
    {"query": "O que causa vibracao excessiva no motor?",                                     "domain": "engine"},
    {"query": "Qual o impacto do ruido acustico alto no motor?",                              "domain": "engine"},
    {"query": "Como a pressao de admissao afeta a condicao do motor?",                        "domain": "engine"},
    # Perguntas multi-domínio (devem buscar nos 3 datasets)
    {"query": "Como prevenir falhas em veiculos de frota com uso intenso?",                   "domain": "multi"},
    {"query": "Quais sensores monitorar para diagnostico preventivo completo?",               "domain": "multi"},
]

SOURCE_LABELS = {
    "vehicle_maintenance": "Manutenção",
    "car_predictive":      "Sensores",
    "engine_fault":        "Motor",
}


def run_retrieval_eval():
    """Avalia apenas a etapa de retrieval (rápido, sem chamar o LLM)."""
    from src.rag.retriever import Retriever

    print("=" * 65)
    print("  AVALIAÇÃO DO RETRIEVER (busca semântica no Milvus)")
    print("=" * 65)

    retriever = Retriever()
    results = []

    for item in EVAL_QUERIES:
        query = item["query"]
        t0 = time.time()
        docs = retriever.retrieve(query)
        latency_ms = (time.time() - t0) * 1000

        scores = [d.score for d in docs]
        sources_hit = {d.source for d in docs}
        sources_str = ", ".join(SOURCE_LABELS.get(s, s) for s in sources_hit)

        results.append({
            "query":         query[:55],
            "domain":        item["domain"],
            "n_docs":        len(docs),
            "max_score":     max(scores) if scores else 0,
            "min_score":     min(scores) if scores else 0,
            "mean_score":    statistics.mean(scores) if scores else 0,
            "sources_hit":   len(sources_hit),
            "sources":       sources_str,
            "latency_ms":    latency_ms,
        })

        status = "✓" if len(sources_hit) >= 2 else ("△" if len(sources_hit) == 1 else "✗")
        print(f"\n  {status} [{item['domain']:11}] {query[:55]}")
        print(f"     docs={len(docs):2}  max_score={max(scores):.3f}  mean={statistics.mean(scores):.3f}  "
              f"fontes=({sources_str})  latência={latency_ms:.0f}ms")

    # ── Summarize ──────────────────────────────────────────────────────────────
    all_scores = [r["mean_score"] for r in results]
    all_latencies = [r["latency_ms"] for r in results]
    multi_source_rate = sum(1 for r in results if r["sources_hit"] >= 2) / len(results)

    print()
    print("=" * 65)
    print("  RESUMO RETRIEVAL")
    print("=" * 65)
    print(f"  Queries testadas:         {len(results)}")
    print(f"  Score médio global:       {statistics.mean(all_scores):.4f}")
    print(f"  Score máximo global:      {max(r['max_score'] for r in results):.4f}")
    print(f"  Score mínimo global:      {min(r['min_score'] for r in results):.4f}")
    print(f"  Taxa multi-source (≥2):   {multi_source_rate:.0%}")
    print(f"  Latência média busca:     {statistics.mean(all_latencies):.0f}ms")
    print(f"  Latência máx busca:       {max(all_latencies):.0f}ms")
    print()

    # Por domínio
    for domain in ["maintenance", "predictive", "engine", "multi"]:
        dr = [r for r in results if r["domain"] == domain]
        if dr:
            avg = statistics.mean(r["mean_score"] for r in dr)
            ms_rate = sum(1 for r in dr if r["sources_hit"] >= 2) / len(dr)
            print(f"  [{domain:11}]  score_médio={avg:.4f}  multi_source={ms_rate:.0%}")

    return results


def run_rag_eval(n_queries: int = 5):
    """Avalia o pipeline RAG completo (retrieval + geração com LLM)."""
    from src.rag.pipeline import RAGPipeline

    print()
    print("=" * 65)
    print("  AVALIAÇÃO RAG COMPLETO (retrieval + LLM)")
    print("=" * 65)
    print(f"  (testando {n_queries} queries — LLM é lento, seja paciente)\n")

    pipeline = RAGPipeline()
    results = []

    for item in EVAL_QUERIES[:n_queries]:
        query = item["query"]
        print(f"  → {query[:60]}")

        t0 = time.time()
        try:
            response = pipeline.query(query, min_score=0.20)
            latency_s = time.time() - t0
            answer_words = len(response.answer.split())
            sources_hit = {s["source"] for s in response.sources}
            scores = [s["score"] for s in response.sources] if response.sources else [0]

            results.append({
                "query":         query[:55],
                "domain":        item["domain"],
                "latency_s":     latency_s,
                "answer_words":  answer_words,
                "n_docs":        response.total_docs_retrieved,
                "mean_score":    statistics.mean(scores),
                "sources_hit":   len(sources_hit),
                "has_answer":    len(response.answer) > 50,
            })
            print(f"     ✓ {answer_words} palavras | {response.total_docs_retrieved} docs | "
                  f"score_médio={statistics.mean(scores):.3f} | {latency_s:.1f}s")

        except Exception as e:
            print(f"     ✗ ERRO: {e}")
            results.append({
                "query": query[:55], "domain": item["domain"],
                "latency_s": 0, "answer_words": 0, "n_docs": 0,
                "mean_score": 0, "sources_hit": 0, "has_answer": False,
            })

    if results:
        ok = [r for r in results if r["has_answer"]]
        print()
        print("=" * 65)
        print("  RESUMO RAG")
        print("=" * 65)
        print(f"  Respostas geradas:        {len(ok)}/{len(results)}")
        if ok:
            print(f"  Latência média total:     {statistics.mean(r['latency_s'] for r in ok):.1f}s")
            print(f"  Tamanho médio resposta:   {statistics.mean(r['answer_words'] for r in ok):.0f} palavras")
            print(f"  Docs usados (média):      {statistics.mean(r['n_docs'] for r in ok):.1f}")
            print(f"  Score médio retrieval:    {statistics.mean(r['mean_score'] for r in ok):.4f}")
            print(f"  Taxa multi-source (≥2):   {sum(1 for r in ok if r['sources_hit'] >= 2)/len(ok):.0%}")

    return results


def run_ml_summary():
    """Exibe resumo das métricas ML já registradas no MLflow."""
    try:
        import mlflow
        import os
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

        print()
        print("=" * 65)
        print("  RESUMO MODELOS ML (MLflow)")
        print("=" * 65)

        experiments = [
            ("AutoPredict-Maintenance",  "maintenance",  "binary"),
            ("AutoPredict-Predictive",   "predictive",   "binary"),
            ("AutoPredict-EngineFault",  "engine_fault", "multiclass"),
        ]

        for exp_name, domain, task in experiments:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                print(f"  [{domain}] Experimento não encontrado no MLflow")
                continue

            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.f1_score DESC"],
                max_results=3,
            )

            if runs.empty:
                print(f"  [{domain}] Sem runs no MLflow")
                continue

            print(f"\n  [{domain}] ({task})")
            for _, row in runs.iterrows():
                model_name = row.get("tags.mlflow.runName", row["run_id"][:8])
                acc  = row.get("metrics.accuracy",  "?")
                f1   = row.get("metrics.f1_score",  "?")
                auc  = row.get("metrics.roc_auc",   "?")
                fmt_acc = f"{acc:.4f}" if isinstance(acc, float) else acc
                fmt_f1  = f"{f1:.4f}"  if isinstance(f1,  float) else f1
                fmt_auc = f"{auc:.4f}" if isinstance(auc, float) else auc
                print(f"    {model_name:<40}  acc={fmt_acc}  f1={fmt_f1}  auc={fmt_auc}")

    except Exception as e:
        print(f"  Erro ao consultar MLflow: {e}")


def print_improvement_analysis(retrieval_results, rag_results):
    """Analisa os pontos fracos e sugere melhorias."""
    print()
    print("=" * 65)
    print("  ANÁLISE DE PONTOS DE MELHORIA")
    print("=" * 65)

    # Scores de retrieval
    mean_scores = [r["mean_score"] for r in retrieval_results]
    global_mean = statistics.mean(mean_scores)
    weak_queries = [r for r in retrieval_results if r["mean_score"] < 0.30]
    single_source = [r for r in retrieval_results if r["sources_hit"] <= 1]

    print(f"\n  [RETRIEVAL]")
    print(f"  Score médio global: {global_mean:.4f}")
    if global_mean < 0.40:
        print("  ⚠ Scores baixos → embeddings ou chunking podem melhorar")
    if weak_queries:
        print(f"  ⚠ {len(weak_queries)} queries com score < 0.30:")
        for r in weak_queries:
            print(f"     - {r['query'][:60]} (score={r['mean_score']:.3f})")
    if single_source:
        print(f"  ⚠ {len(single_source)} queries recuperaram de apenas 1 dataset (sem diversidade)")

    # Análise RAG
    if rag_results:
        ok_rag = [r for r in rag_results if r.get("has_answer")]
        print(f"\n  [RAG COMPLETO]")
        if ok_rag:
            avg_lat = statistics.mean(r["latency_s"] for r in ok_rag)
            avg_words = statistics.mean(r["answer_words"] for r in ok_rag)
            print(f"  Latência média: {avg_lat:.1f}s | Tamanho médio: {avg_words:.0f} palavras")
            if avg_lat > 90:
                print("  ⚠ Latência alta → llama3.2:3b rodando em CPU (esperado)")
            if avg_words < 100:
                print("  ⚠ Respostas curtas → possível timeout ou prompt muito longo")

    # Sugestões concretas (baseadas no estado atual do sistema)
    print()
    print("  [ESTADO DAS OTIMIZAÇÕES]")
    print("  ✓ Embedding multilingual (paraphrase-multilingual-MiniLM-L12-v2)")
    print("  ✓ top_k_per_collection = 5 (era 3)")
    print("  ✓ MMR re-ranking com lambda=0.65")
    print("  ✓ Few-shot prompt no gerador")
    print("  ✓ Correção double-balancing no treinamento ML")
    print("  ✓ num_predict = 800 (balanceia qualidade vs latência no CPU)")

    print()
    print("  [SUGESTÕES DE MELHORIA FUTURAS]")
    sugg = []

    if global_mean < 0.45:
        sugg.append(("Chunking adaptativo",
                      "Reduzir chunks de ~200 para 100-150 tokens com 20% overlap "
                      "para aumentar precisão semântica dos embeddings"))
    sugg.append(("GPU para Ollama",
                  "Adicionar GPU ao container Ollama — latência cairia de ~100s para <5s, "
                  "permitindo num_predict=1500 sem timeout"))
    sugg.append(("Dados predictive",
                  "Dataset cars_hyundai tem apenas 1.100 registros — coletar mais dados "
                  "ou fazer data augmentation (SMOTE) para melhorar F1=0.556"))
    sugg.append(("Engine fault dataset",
                  "Correlação das features com target < 0.02 — o dataset não tem sinal preditivo. "
                  "Substituir por dataset real de sensores de motor para obter F1 > 0.50"))
    sugg.append(("Cache de embeddings",
                  "Cachear embeddings de queries frequentes no Redis/in-memory "
                  "para reduzir latência de busca de ~500ms para <10ms"))
    sugg.append(("Modelo de embedding maior",
                  "Testar 'paraphrase-multilingual-mpnet-base-v2' (768 dims) para "
                  "scores de similaridade mais altos (potencial +10-15% nos scores)"))

    for i, (titulo, descricao) in enumerate(sugg, 1):
        print(f"\n  {i}. {titulo}")
        print(f"     {descricao}")

    print()


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    n_rag = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          AutoPredict AI — Avaliação de Métricas              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    retrieval_results = run_retrieval_eval()

    if mode == "full":
        rag_results = run_rag_eval(n_queries=n_rag)
    else:
        rag_results = []

    run_ml_summary()
    print_improvement_analysis(retrieval_results, rag_results if rag_results else retrieval_results)

    print("=" * 65)
    print("  Avaliação concluída.")
    print("=" * 65)
    print()
