# AutoPredict AI — Checklist de Desenvolvimento
> Baseado nas 12 Sprints do projeto. Siga as tasks **em ordem**, uma por vez.
> Legenda: `[ ]` pendente · `[x]` concluído · `[~]` parcialmente feito · `[!]` bloqueante

---

## SPRINT 1 — Definição do Produto ✅
> Entregável: Product Backlog inicial

- [x] S1-01 · Escolher domínio do projeto (manutenção preditiva automotiva)
- [x] S1-02 · Definir empresa fictícia (AutoPredict AI)
- [x] S1-03 · Definir problema de negócio (falhas inesperadas em frotas)
- [x] S1-04 · Levantar requisitos funcionais (RF1–RF5)
- [x] S1-05 · Levantar requisitos não funcionais (RNF1–RNF5)
- [x] S1-06 · Definir papéis Scrum (PO, Scrum Master, Devs)
- [x] S1-07 · Selecionar datasets públicos (3 datasets do Kaggle)
- [x] S1-08 · Criar Product Backlog inicial no README.md
- [x] S1-09 · Desenhar arquitetura conceitual (diagrama ASCII no README)

---

## SPRINT 2 — Arquitetura e Infraestrutura Base
> Entregável: Diagrama arquitetural + Docker Compose funcional

### 2A · Arquitetura
- [x] S2-01 · Definir arquitetura geral (MinIO + PostgreSQL + Milvus + Ollama + FastAPI + Gradio)
- [x] S2-02 · Documentar arquitetura no README (diagrama ASCII)
- [ ] S2-03 · Criar arquivo `.env` com todas as variáveis de ambiente
  ```
  # Criar o arquivo .env na raiz do projeto com:
  MINIO_USER=minioadmin
  MINIO_PASSWORD=minioadmin123
  POSTGRES_USER=autopredict
  POSTGRES_PASSWORD=autopredict123
  POSTGRES_DB=autopredict
  OLLAMA_MODEL=llama3.2:3b
  ```

### 2B · Docker Compose
- [x] S2-04 · `docker-compose.yml` com todos os serviços criado
- [x] S2-05 · `name: autopredict` adicionado para isolar do ambiente de trabalho
- [ ] S2-06 · Executar `docker compose up -d` e aguardar todos os health checks passarem
- [ ] S2-07 · Verificar MinIO acessível: abrir `http://localhost:9001` (login: minioadmin / minioadmin123)
- [ ] S2-08 · Verificar PostgreSQL respondendo: `docker exec autopredict-postgres pg_isready`
- [ ] S2-09 · Verificar Milvus respondendo: `docker exec autopredict-milvus curl -s http://localhost:9091/healthz`
- [ ] S2-10 · Verificar Ollama respondendo: `curl http://localhost:11434`

### 2C · Setup MinIO (buckets)
- [ ] S2-11 · No console MinIO (`http://localhost:9001`), criar bucket `bronze`
- [ ] S2-12 · Criar bucket `silver`
- [ ] S2-13 · Criar bucket `gold`

### 2D · Setup PostgreSQL (schema)
- [ ] S2-14 · Build da imagem da API: `docker compose build api`
- [ ] S2-15 · Inicializar tabelas do PostgreSQL executando:
  ```bash
  docker exec autopredict-api python -c "from src.database.postgres_client import PostgresClient; PostgresClient()"
  ```

### 2E · Build dos containers customizados
- [x] S2-16 · `Dockerfile.api` criado
- [x] S2-17 · `Dockerfile.frontend` criado
- [ ] S2-18 · `docker compose build` sem erros
- [ ] S2-19 · `docker compose up -d` com todos os containers em estado `running`

---

## SPRINT 3 — Governança e Medallion
> Entregável: Dados organizados nas 3 camadas + documentação de governança

### 3A · Camada Bronze (ingestão dos CSVs brutos)
- [x] S3-01 · `src/data_pipeline/bronze.py` criado
- [x] S3-02 · Nomes dos arquivos corrigidos para os CSVs reais:
  - `vehicle_maintenance_data.csv`
  - `cars_hyundai.csv`
  - `engine_fault_detection_dataset.csv`
- [ ] S3-03 · Confirmar que os 3 CSVs estão em `data/`
- [ ] S3-04 · Rodar ingestão Bronze:
  ```bash
  docker exec autopredict-api python -c "
  from src.data_pipeline.bronze import ingest_to_bronze
  print(ingest_to_bronze('/app/data'))
  "
  ```
- [ ] S3-05 · Verificar no console MinIO que os 3 arquivos aparecem no bucket `bronze`

### 3B · Camada Silver (limpeza)
- [x] S3-06 · `src/data_pipeline/silver.py` criado
- [x] S3-07 · Sanitização de colunas especiais (`°C`, `(mm)`, `(PSI)`) implementada com regex
- [x] S3-08 · Detecção de `anomaly_indication` adicionada (cars_hyundai)
- [x] S3-09 · Detecção de `engine_condition` adicionada (engine_fault)
- [ ] S3-10 · Rodar pipeline Silver:
  ```bash
  docker exec autopredict-api python -c "
  from src.data_pipeline.silver import process_to_silver
  print(process_to_silver())
  "
  ```
- [ ] S3-11 · Verificar no bucket `silver` os 3 arquivos `silver_*.csv`
- [ ] S3-12 · Inspecionar visualmente um silver CSV e confirmar colunas limpas

### 3C · Camada Gold (chunking + embeddings + Milvus)
- [x] S3-13 · `src/data_pipeline/gold.py` criado
- [x] S3-14 · Funções de chunking reescritas com colunas reais dos 3 datasets
- [x] S3-15 · Silver filenames corrigidos no `GOLD_CONFIG`
- [x] S3-16 · Metadados de key_cols atualizados para as colunas reais
- [ ] S3-17 · Baixar modelo LLM no Ollama:
  ```bash
  docker exec autopredict-ollama ollama pull llama3.2:3b
  ```
- [ ] S3-18 · Rodar pipeline Gold (gera embeddings e indexa no Milvus):
  ```bash
  docker exec autopredict-api python /app/src/data_pipeline/run_pipeline.py
  ```
- [ ] S3-19 · Verificar no bucket `gold` os 3 arquivos `gold_*.csv`
- [ ] S3-20 · Verificar no PostgreSQL que `ingestion_log` tem registros de todos os layers
- [ ] S3-21 · Verificar que as 3 coleções Milvus foram criadas e têm documentos

### 3D · Documentação de Governança
- [ ] S3-22 · Criar `data/README.md` explicando estrutura dos 3 CSVs e colunas
- [ ] S3-23 · Atualizar README.md: marcar Sprint 3 como concluída no Roadmap

---

## SPRINT 4 — Modelagem e Treinamento de ML + MLflow (AC1) ⚠️ CRÍTICO
> Entregável: Pipeline de treino funcional + experimentos registrados no MLflow
> **ATENÇÃO: Esta sprint não tem nenhum código ainda. É a AC1 (nota).**

### 4A · Setup MLflow
- [x] S4-01 · Adicionar serviço `mlflow` ao `docker-compose.yml`
- [x] S4-02 · Adicionar `mlflow`, `scikit-learn`, `xgboost` e `boto3` ao `requirements.txt`
- [x] S4-03 · Bucket `mlflow-artifacts` criado automaticamente pelo `MinIOClient`
- [ ] S4-04 · Rebuildar containers: `docker compose up -d --build` (depende S2-06)
- [ ] S4-05 · Verificar MLflow acessível em `http://localhost:5001`

### 4B · Feature Engineering
- [x] S4-06 · Criar `src/ml/__init__.py`
- [x] S4-07 · Criar `src/ml/feature_engineering.py`
  - Problema 1: **Classificação binária** — prever `need_maintenance` (vehicle_maintenance)
  - Problema 2: **Classificação binária** — prever `anomaly_indication` (cars_hyundai)
  - Problema 3: **Classificação multiclasse** — prever `engine_condition` 0/1/2 (engine_fault)
- [x] S4-08 · `load_dataset()` com fallback MinIO → disco
- [x] S4-09 · `prepare_maintenance_features()` — encode ordinárias, features derivadas
- [x] S4-10 · `prepare_predictive_features()` — features de sensor + thresholds
- [x] S4-11 · `prepare_engine_features()` — features de vibração + temperatura
- [x] S4-12 · `split_and_scale()` — train/test 80/20 + StandardScaler
- [x] S4-13 · `get_all_datasets()` — carrega e prepara os 3 datasets de uma vez

### 4C · Scripts de Treinamento
- [x] S4-14 · Criar `src/ml/train.py`
- [x] S4-15 · Implementar treinamento **Logistic Regression** (baseline linear)
- [x] S4-16 · Implementar treinamento **Random Forest**
- [x] S4-17 · Implementar treinamento **XGBoost**
- [x] S4-18 · Calcular métricas: accuracy, precision, recall, F1, ROC-AUC
- [x] S4-19 · `mlflow.log_metric`, `mlflow.log_param` por run
- [x] S4-20 · Registrar modelo no MLflow Model Registry (`mlflow.sklearn.log_model`)
- [x] S4-21 · Salvar `classification_report.txt` como artefato de cada run
- [x] S4-22 · Criar `src/ml/predict.py` — `VehicleHealthPredictor` (combina os 3 modelos)
  - Combina os 3 riscos com pesos: maintenance=0.25, sensores=0.35, motor=0.40
  - Retorna `VehicleHealthResult` com `vehicle_health_index` (0.0=perfeito → 1.0=crítico)

### 4D · Integração MLflow
- [x] S4-23 · Experimento `AutoPredict-Maintenance` definido no código
- [x] S4-24 · Experimento `AutoPredict-Predictive` definido no código
- [x] S4-25 · Experimento `AutoPredict-EngineFault` definido no código
- [ ] S4-26 · Executar todos os treinamentos (depende S3-18 + S4-05):
  ```bash
  docker exec autopredict-api python /app/src/ml/train.py
  ```
- [ ] S4-27 · Verificar no MLflow UI (`http://localhost:5001`) os 3 experimentos com runs
- [ ] S4-28 · Comparar modelos no MLflow e identificar o melhor por dataset
- [ ] S4-29 · Registrar o melhor modelo de cada experimento no Model Registry

### 4E · Documentação AC1
- [ ] S4-26 · Atualizar README.md: seção Sprint 4 com decisões de modelagem
- [ ] S4-27 · Documentar no README qual modelo venceu em cada dataset e as métricas

---

## SPRINT 5 — Pipeline de Embeddings
> Entregável: Index vetorial funcional + processo automatizado

### 5A · Setup Milvus (depende Sprint 2)
- [x] S5-01 · `src/database/milvus_client.py` criado com schema
- [x] S5-02 · Schema com campos: `id`, `text`, `source`, `metadata`, `embedding` (384 dims)
- [x] S5-03 · 3 coleções definidas: `vehicle_maintenance`, `car_predictive`, `engine_fault`
- [ ] S5-04 · Verificar que Milvus está rodando (depende S2-06)
- [ ] S5-05 · Testar conexão com Milvus:
  ```bash
  docker exec autopredict-api python -c "
  from src.database.milvus_client import MilvusClient
  c = MilvusClient(); print('Milvus OK')
  "
  ```

### 5B · Geração de Embeddings
- [x] S5-06 · `src/embeddings/embedder.py` criado com `sentence-transformers`
- [x] S5-07 · Modelo `all-MiniLM-L6-v2` configurado (384 dimensões)
- [x] S5-08 · Método `embed_batch()` implementado
- [ ] S5-09 · Testar embedder isoladamente:
  ```bash
  docker exec autopredict-api python -c "
  from src.embeddings.embedder import get_embedder
  e = get_embedder()
  v = e.embed_batch(['motor superaquecido'])
  print('Shape:', len(v[0]), 'dims — OK')
  "
  ```

### 5C · Indexação Vetorial
- [x] S5-10 · `process_to_gold()` em `gold.py` gera chunks → embeddings → insere no Milvus
- [ ] S5-11 · Executar gold pipeline completo (depende S3-18)
- [ ] S5-12 · Verificar contagem de documentos em cada coleção:
  ```bash
  docker exec autopredict-api python -c "
  from src.database.milvus_client import MilvusClient
  c = MilvusClient()
  for col in ['vehicle_maintenance','car_predictive','engine_fault']:
      print(col, c.get_collection_stats(col))
  "
  ```

### 5D · Integração Ollama
- [ ] S5-13 · Confirmar que o modelo `llama3.2:3b` foi baixado (depende S3-17)
- [ ] S5-14 · Testar geração de texto com Ollama:
  ```bash
  curl http://localhost:11434/api/generate -d '{
    "model": "llama3.2:3b",
    "prompt": "O que causa superaquecimento do motor?",
    "stream": false
  }'
  ```

### 5E · Teste End-to-End do Pipeline RAG
- [x] S5-15 · `src/rag/retriever.py` criado (busca multi-coleção)
- [x] S5-16 · `src/rag/generator.py` criado (monta prompt + chama Ollama)
- [x] S5-17 · `src/rag/pipeline.py` criado (orquestra retriever + generator)
- [ ] S5-18 · Testar pipeline RAG completo:
  ```bash
  docker exec autopredict-api python -c "
  from src.rag.pipeline import RAGPipeline
  p = RAGPipeline()
  r = p.query('Quais falhas são comuns em veículos com mais de 80.000km?')
  print(r.answer)
  print('Fontes:', r.total_docs_retrieved)
  "
  ```
- [ ] S5-19 · Verificar que a resposta usa contexto dos 3 datasets
- [ ] S5-20 · Ajustar `min_score` em `config.py` se necessário (padrão: 0.25)

### 5F · API e Frontend
- [x] S5-21 · `src/api/main.py` criado com FastAPI
- [x] S5-22 · Rota `POST /chat` implementada
- [x] S5-23 · Rota `GET /health` implementada
- [x] S5-24 · `src/frontend/app.py` criado com Gradio
- [ ] S5-25 · Acessar Swagger em `http://localhost:8000/docs` e testar `/chat`
- [ ] S5-26 · Acessar frontend em `http://localhost:7860` e fazer uma pergunta
- [ ] S5-27 · Verificar que fontes são exibidas corretamente no chat

---

## TAREFAS TRANSVERSAIS (fazer durante as sprints)

### Qualidade de Código
- [ ] T-01 · Instalar dependências localmente para análise: `pip install -r requirements.txt`
- [ ] T-02 · Verificar erros de importação nos módulos principais
- [ ] T-03 · Testar `src/config.py` carrega `.env` corretamente

### Git / Versionamento
- [ ] T-04 · Criar `.gitignore` adequado (já existe em `data/`, verificar na raiz)
- [ ] T-05 · Fazer commits separados por sprint (mensagens descritivas)
- [ ] T-06 · Nunca commitar o arquivo `.env` (dados sensíveis)

---

## RESUMO DE STATUS POR SPRINT

| Sprint | Status | Bloqueante principal |
|--------|--------|----------------------|
| Sprint 1 | ✅ Concluída | — |
| Sprint 2 | 🔄 Em andamento | Executar `docker compose up` |
| Sprint 3 | 🔄 Em andamento | Depende Sprint 2 + dados no MinIO |
| Sprint 4 | ❌ Não iniciada | **MLflow + código ML inexistente** |
| Sprint 5 | 🔄 Em andamento | Depende Sprints 2, 3 e 4 |

---

## ORDEM DE EXECUÇÃO RECOMENDADA

```
S2-03 (criar .env)
  → S2-06 (docker compose up)
    → S2-07~S2-13 (verificar serviços + criar buckets)
      → S2-14~S2-19 (build + postgres init)
        → S3-03~S3-05 (Bronze)
          → S3-10~S3-12 (Silver)
            → S3-17 (baixar LLM)
              → S3-18~S3-21 (Gold + Milvus)
                → S4-01~S4-05 (MLflow setup)  ← SPRINT 4 COMEÇA AQUI
                  → S4-06~S4-25 (ML training)
                    → S5-04~S5-27 (Embeddings + RAG + API + Frontend)
```
