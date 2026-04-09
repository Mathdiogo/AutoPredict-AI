# AutoPredict AI 🚗🤖

**Plataforma de Diagnóstico Automotivo Preditivo com RAG + LLM Local**

O AutoPredict AI é um chatbot especializado em manutenção de veículos. Ele usa **RAG (Retrieval-Augmented Generation)** para buscar informações em 3 bases de dados reais e gerar respostas fundamentadas usando um LLM rodando 100% localmente (sem custos de API).

> Para documentação de produto, backlog e requisitos, veja [PROJECT_SPEC.md](PROJECT_SPEC.md).

---

## Sumário

1. [Pré-requisitos](#1-pré-requisitos)
2. [Instalação](#2-instalação)
3. [Como Rodar](#3-como-rodar)
4. [Primeiros Passos (setup inicial)](#4-primeiros-passos-setup-inicial)
5. [Como Usar](#5-como-usar)
6. [Serviços e Portas](#6-serviços-e-portas)
7. [Arquitetura Completa](#7-arquitetura-completa)
8. [Product Backlog Completo](#8-product-backlog-completo)
9. [Governança de Dados](#9-governança-de-dados)
10. [Comandos Úteis](#10-comandos-úteis)
11. [Solução de Problemas](#11-solução-de-problemas)

---

## 1. Pré-requisitos

Antes de começar, você precisa ter instalado:

| Ferramenta | Versão mínima | Download |
|---|---|---|
| Docker Desktop | 24.x | https://www.docker.com/products/docker-desktop |
| Docker Compose | 2.x (incluído no Desktop) | — |
| Git | qualquer | https://git-scm.com |

> **Recursos de hardware recomendados:** 16 GB RAM, 20 GB de espaço em disco.
> O LLM (llama3.2:3b) roda em CPU, então mais RAM = melhor desempenho.

Verifique se tudo está instalado:

```bash
docker --version        # Docker version 24.x.x
docker compose version  # Docker Compose version v2.x.x
```

---

## 2. Instalação

### 2.1 Clone o repositório

```bash
git clone https://github.com/Mathdiogo/AutoPredict-AI.git
cd AutoPredict-AI
```

### 2.2 Estrutura de pastas necessária

Crie a pasta `data/` com os datasets brutos (Bronze):

```
AutoPredict-AI/
├── data/
│   ├── vehicle_maintenance_data.csv   ← Dataset 1 (Kaggle)
│   ├── cars_hyundai.csv               ← Dataset 2 (Kaggle)
│   └── engine_fault_detection_dataset.csv  ← Dataset 3 (Kaggle)
```

**Links dos datasets:**
- [Vehicle Maintenance Data](https://www.kaggle.com/datasets/chavindudulaj/vehicle-maintenance-data)
- [Car Predictive Maintenance](https://www.kaggle.com/datasets/pragyanaianddsschool/car-predictive-maintenance-data)
- [Engine Fault Detection](https://www.kaggle.com/datasets/ziya07/engine-fault-detection-data)

> Se a pasta `data/` não existir, o pipeline de ingestão irá falhar com `FileNotFoundError`.

---

## 3. Como Rodar

### 3.1 Subir todos os serviços

```bash
docker compose up -d
```

Isso vai iniciar **8 containers**:

| Container | Função |
|---|---|
| `autopredict-api` | API REST (FastAPI) |
| `autopredict-frontend` | Interface web (Gradio) |
| `autopredict-ollama` | LLM local (llama3.2:3b) |
| `autopredict-milvus` | Banco vetorial (embeddings) |
| `autopredict-postgres` | Banco relacional (metadados) |
| `autopredict-minio` | Data Lake (arquivos CSV) |
| `autopredict-mlflow` | Experimentos de ML |
| `autopredict-etcd` | Dependência interna do Milvus |

Aguarde ~1 minuto até todos os serviços ficarem saudáveis:

```bash
docker compose ps
```

Todos devem exibir `Up` ou `healthy`.

### 3.2 Parar os serviços

```bash
docker compose down
```

Para parar **e apagar os volumes** (dados serão perdidos):

```bash
docker compose down -v
```

---

## 4. Primeiros Passos (setup inicial)

Na **primeira execução**, você precisa rodar o pipeline de dados e baixar o modelo LLM. Isso só precisa ser feito uma vez.

### Passo 1 — Baixar o modelo LLM

```bash
docker exec autopredict-ollama ollama pull llama3.2:3b
```

> O download é de ~2 GB. Aguarde a conclusão antes de continuar.

### Passo 2 — Executar o pipeline de dados (Bronze → Silver → Gold)

O pipeline processa os CSVs brutos, limpa os dados e gera os embeddings no Milvus:

```bash
docker exec -w /app autopredict-api python -m src.data_pipeline.run_pipeline
```

Isso executa em sequência:
1. **Bronze** — lê os CSVs de `data/` e salva no MinIO
2. **Silver** — limpa e normaliza os dados
3. **Gold** — gera embeddings e indexa no Milvus (11.100 documentos)

> Tempo estimado: ~5-10 minutos (inclui download do modelo de embedding na primeira vez).

### Passo 3 — Treinar os modelos de ML

```bash
docker exec -w /app autopredict-api python -m src.ml.train
```

Treina 3 modelos (Logistic Regression, Random Forest, XGBoost) para cada um dos 3 datasets e registra no MLflow.

> Tempo estimado: ~2-3 minutos.

### Verificar se está tudo ok

```bash
docker exec -w /app autopredict-api python -c "
from src.database.milvus_client import MilvusClient
c = MilvusClient()
for col in ['vehicle_maintenance', 'car_predictive', 'engine_fault']:
    print(col, c.get_count(col))
"
```

Saída esperada:
```
vehicle_maintenance 5000
car_predictive 1100
engine_fault 5000
```

---

## 5. Como Usar

### 5.1 Interface Web (recomendado)

Acesse o chat em: **http://localhost:7860**

A interface permite:
- Digitar perguntas sobre manutenção e diagnóstico de veículos
- Ver as respostas sendo geradas em tempo real (streaming)
- Ativar "Mostrar documentos usados como contexto" para ver as fontes

**Exemplos de perguntas:**

```
Quais são as causas mais comuns de superaquecimento do motor?
Qual a pressão ideal dos pneus?
Como saber se o freio precisa de manutenção?
O que pode causar vibração excessiva no motor?
Meu carro tem 80.000km, o que verificar preventivamente?
Qual a espessura mínima da pastilha de freio antes de trocar?
```

### 5.2 API REST

A API está disponível em **http://localhost:8000**.

Documentação interativa (Swagger): **http://localhost:8000/docs**

**Fazer uma pergunta via POST:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "O que causa falha no motor?", "min_score": 0.20}'
```

Resposta:
```json
{
  "answer": "Com base nos dados de sensores...",
  "sources": [...],
  "total_docs_retrieved": 15,
  "model": "llama3.2:3b"
}
```

**Streaming (tokens em tempo real) via GET:**

```bash
curl "http://localhost:8000/chat/stream?question=Qual+a+pressao+ideal+dos+pneus"
```

**Verificar status dos serviços:**

```bash
curl http://localhost:8000/health
```

### 5.3 MLflow (experimentos de ML)

Acesse o painel de experimentos em: **http://localhost:5001**

Lá você encontra:
- Métricas de cada modelo (Accuracy, F1, AUC)
- Comparação entre modelos por dataset
- Artefatos (relatórios de classificação)

### 5.4 MinIO (Data Lake)

Acesse o console em: **http://localhost:9001**

Credenciais padrão:
- **Usuário:** `minioadmin`
- **Senha:** `minioadmin`

Os dados estão organizados em buckets:
- `bronze/` — CSVs brutos originais
- `silver/` — dados limpos e normalizados
- `gold/` — dados processados prontos para o modelo

---

## 6. Serviços e Portas

| Serviço | URL | Credenciais |
|---|---|---|
| Interface Web (Gradio) | http://localhost:7860 | — |
| API REST | http://localhost:8000 | — |
| API Docs (Swagger) | http://localhost:8000/docs | — |
| MLflow | http://localhost:5001 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Milvus | localhost:19530 | — (interno) |
| PostgreSQL | localhost:5432 | postgres / postgres |
| Ollama | http://localhost:11434 | — (interno) |

---

## 7. Arquitetura Completa

### 7.1 Visão Geral — Todos os Serviços

```
╔══════════════════════════════════════════════════════════════════════╗
║                        CAMADA DE APRESENTAÇÃO                       ║
║                                                                      ║
║   ┌──────────────────────────────────────────────────────────────┐  ║
║   │         Frontend — Gradio  (autopredict-frontend :7860)      │  ║
║   │   Chat em tempo real · Streaming · Toggle de fontes          │  ║
║   └──────────────────────────┬───────────────────────────────────┘  ║
╚═════════════════════════════╪════════════════════════════════════════╝
                              │ HTTP REST
╔═════════════════════════════╪════════════════════════════════════════╗
║                        CAMADA DE API                                ║
║                             │                                        ║
║   ┌──────────────────────────▼───────────────────────────────────┐  ║
║   │           API REST — FastAPI  (autopredict-api :8000)        │  ║
║   │   POST /chat  ·  GET /chat/stream  ·  GET /health            │  ║
║   └──────────┬───────────────────────────────────┬───────────────┘  ║
╚══════════════╪═══════════════════════════════════╪════════════════════╝
               │                                   │
╔══════════════╪═══════════════════════════════════╪════════════════════╗
║              │      CAMADA DE IA / RAG           │                    ║
║   ┌──────────▼──────────────┐       ┌────────────▼─────────────────┐ ║
║   │   Retriever             │       │   Generator                  │ ║
║   │   ─────────             │       │   ─────────                  │ ║
║   │   1. Gera embedding     │       │   1. Monta prompt few-shot   │ ║
║   │   2. Busca no Milvus    │       │   2. Envia ao Ollama         │ ║
║   │   3. MMR reranking      │       │   3. Retorna stream tokens   │ ║
║   └──────────┬──────────────┘       └────────────┬─────────────────┘ ║
║              │                                   │                    ║
║   ┌──────────▼──────────────┐       ┌────────────▼─────────────────┐ ║
║   │   Embedder              │       │   Ollama  (:11434)           │ ║
║   │   sentence-transformers │       │   modelo: llama3.2:3b        │ ║
║   │   paraphrase-MiniLM     │       │   CPU-only · 2GB             │ ║
║   └─────────────────────────┘       └──────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════╗
║                         CAMADA DE DADOS                              ║
║                                                                       ║
║  ┌─────────────────────────────────────────────────────────────────┐ ║
║  │   Milvus — Banco Vetorial  (autopredict-milvus :19530)          │ ║
║  │   ├── vehicle_maintenance   5.000 docs                          │ ║
║  │   ├── car_predictive        1.100 docs                          │ ║
║  │   └── engine_fault          5.000 docs                          │ ║
║  └──────────────────────┬──────────────────────────────────────────┘ ║
║                         │ depende de                                  ║
║  ┌──────────────────────▼──────────┐  ┌───────────────────────────┐  ║
║  │  etcd  (autopredict-etcd)       │  │  MinIO  (:9000 / :9001)   │  ║
║  │  Configuração distribuída       │  │  Buckets: bronze/         │  ║
║  │  do Milvus (interno)            │  │           silver/         │  ║
║  └─────────────────────────────────┘  │           gold/           │  ║
║                                       └───────────────────────────┘  ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │  PostgreSQL  (autopredict-postgres :5432)                        │ ║
║  │  Logs de ingestão · Metadados das runs · Auditoria               │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
╚════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════╗
║                         CAMADA DE ML / OBSERVABILIDADE               ║
║                                                                        ║
║  ┌───────────────────────────────────────────────────────────────┐    ║
║  │  MLflow  (autopredict-mlflow :5001)                           │    ║
║  │  3 experimentos · 9 modelos · Métricas: Acc, F1, AUC          │    ║
║  │  Artefatos: classification_report.json, model.pkl             │    ║
║  └───────────────────────────────────────────────────────────────┘    ║
╚═════════════════════════════════════════════════════════════════════════╝
```

### 7.2 Pipeline de Dados — Padrão Medallion

```
  data/ (CSV brutos)
       │
       ▼
  ┌─────────────────────────────────────────────────────┐
  │  BRONZE  — Ingestão bruta                          │
  │  Lê os 3 CSVs e envia ao MinIO sem transformação   │
  │  • vehicle_maintenance_data.csv  → 50.000 linhas   │
  │  • cars_hyundai.csv              →  1.100 linhas   │
  │  • engine_fault_detection.csv    → 10.000 linhas   │
  └──────────────────────┬──────────────────────────────┘
                         │ MinIO: bucket bronze/
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  SILVER  — Limpeza e padronização                  │
  │  • Remove nulos e duplicatas                       │
  │  • Normaliza nomes de colunas                      │
  │  • Converte tipos de dados                         │
  │  • Limita a 5.000 linhas por dataset               │
  └──────────────────────┬──────────────────────────────┘
                         │ MinIO: bucket silver/
                         ▼
  ┌─────────────────────────────────────────────────────┐
  │  GOLD  — Embeddings + Indexação                    │
  │  • Gera texto descritivo por linha                 │
  │  • Vetoriza com paraphrase-MiniLM (384 dims)       │
  │  • Indexa 11.100 documentos no Milvus              │
  │  • Salva CSV final no MinIO bucket gold/           │
  └─────────────────────────────────────────────────────┘
                         │
                         ▼
              Milvus (pronto para busca semântica)
```

### 7.3 Fluxo de uma Pergunta no RAG

```
  Usuário: "O que causa superaquecimento do motor?"
       │
       ▼
  [1] Embedding da pergunta (paraphrase-MiniLM → vetor 384d)
       │
       ▼
  [2] Busca por similaridade coseno no Milvus (top-15 docs)
       │
       ▼
  [3] MMR reranking (diversidade + relevância)
       │
       ▼
  [4] Montagem do prompt com few-shot + contexto dos docs
       │
       ▼
  [5] Ollama (llama3.2:3b) gera resposta via streaming
       │
       ▼
  Resposta fundamentada com citação das fontes
```

### 7.4 Modelos de Machine Learning

```
  MinIO Silver (dados limpos)
       │
       ▼
  Feature Engineering (src/ml/feature_engineering.py)
       │
       ├── Dataset: vehicle_maintenance  (40k treino / 10k teste)
       │       ├── LogisticRegression   → Acc=0.84  F1=0.90  AUC=0.94
       │       ├── RandomForest         → Acc=0.97  F1=0.98  AUC=1.00
       │       └── XGBoost ★ melhor    → Acc=1.00  F1=1.00  AUC=1.00
       │
       ├── Dataset: car_predictive      (880 treino / 220 teste)
       │       ├── LogisticRegression   → F1=0.52  AUC=0.52
       │       ├── RandomForest ★       → F1=0.56  AUC=0.56
       │       └── XGBoost              → F1=0.54  AUC=0.57
       │
       └── Dataset: engine_fault        (8k treino / 2k teste)
               ├── LogisticRegression
               ├── RandomForest
               └── XGBoost
                       │
                       ▼
              MLflow — 9 modelos registrados com artefatos
```

---

## 8. Product Backlog Completo

### Metodologia Scrum

| Papel | Responsabilidade |
|---|---|
| Product Owner | Define e prioriza o backlog |
| Scrum Master | Garante o processo e remove impedimentos |
| Dev Team | Desenvolvimento, dados e IA |

Sprints de **2 semanas**. Critério de aceite: funcionalidade demonstrável e testada.

---

### Sprint 1 — Definição do Produto ✅

| ID | Item | Prioridade | Status |
|---|---|---|---|
| S1-01 | Definir domínio do projeto (manutenção preditiva) | Alta | ✅ Concluído |
| S1-02 | Definir problema de negócio | Alta | ✅ Concluído |
| S1-03 | Selecionar 3 datasets públicos do Kaggle | Alta | ✅ Concluído |
| S1-04 | Definir arquitetura inicial (Medallion + RAG) | Alta | ✅ Concluído |
| S1-05 | Configurar repositório Git e estrutura de pastas | Média | ✅ Concluído |
| S1-06 | Criar PROJECT_SPEC.md com requisitos funcionais e não funcionais | Média | ✅ Concluído |
| S1-07 | Definir papéis Scrum e equipe | Média | ✅ Concluído |
| S1-08 | Criar README.md inicial | Baixa | ✅ Concluído |

---

### Sprint 2 — Infraestrutura Base ✅

| ID | Item | Prioridade | Status |
|---|---|---|---|
| S2-01 | Criar `docker-compose.yml` com todos os serviços | Alta | ✅ Concluído |
| S2-02 | Configurar MinIO (Data Lake S3-compatível) | Alta | ✅ Concluído |
| S2-03 | Configurar PostgreSQL (metadados e logs) | Alta | ✅ Concluído |
| S2-04 | Configurar Milvus + etcd (banco vetorial) | Alta | ✅ Concluído |
| S2-05 | Configurar Ollama (LLM local) | Alta | ✅ Concluído |
| S2-06 | Criar `Dockerfile.api` para o backend | Média | ✅ Concluído |
| S2-07 | Criar `Dockerfile.frontend` para o Gradio | Média | ✅ Concluído |
| S2-08 | Criar `requirements.txt` com todas as dependências | Média | ✅ Concluído |
| S2-09 | Implementar clientes de banco (`minio_client.py`, `postgres_client.py`, `milvus_client.py`) | Média | ✅ Concluído |
| S2-10 | Criar `config.py` com variáveis de ambiente | Baixa | ✅ Concluído |

---

### Sprint 3 — Pipeline de Dados (Medallion) ✅

| ID | Item | Prioridade | Status |
|---|---|---|---|
| S3-01 | Implementar camada Bronze (`bronze.py`) — ingestão dos CSVs no MinIO | Alta | ✅ Concluído |
| S3-02 | Implementar camada Silver (`silver.py`) — limpeza e normalização | Alta | ✅ Concluído |
| S3-03 | Implementar camada Gold (`gold.py`) — geração de embeddings e indexação no Milvus | Alta | ✅ Concluído |
| S3-04 | Criar `run_pipeline.py` para execução sequencial Bronze→Silver→Gold | Alta | ✅ Concluído |
| S3-05 | Implementar `embedder.py` com sentence-transformers (paraphrase-MiniLM) | Alta | ✅ Concluído |
| S3-06 | Indexar 11.100 documentos no Milvus (5k + 1.1k + 5k) | Alta | ✅ Concluído |
| S3-07 | Registrar logs de ingestão no PostgreSQL | Média | ✅ Concluído |
| S3-08 | Validar contagem de documentos por coleção | Média | ✅ Concluído |

---

### Sprint 4 — Modelos de ML + MLflow ✅

| ID | Item | Prioridade | Status |
|---|---|---|---|
| S4-01 | Implementar `feature_engineering.py` para os 3 datasets | Alta | ✅ Concluído |
| S4-02 | Implementar `train.py` com Logistic Regression, Random Forest e XGBoost | Alta | ✅ Concluído |
| S4-03 | Integrar MLflow tracking (métricas, parâmetros, artefatos) | Alta | ✅ Concluído |
| S4-04 | Registrar 9 modelos no MLflow Model Registry | Alta | ✅ Concluído |
| S4-05 | Implementar `predict.py` para inferência | Média | ✅ Concluído |
| S4-06 | Configurar MLflow server com SQLite e artifacts local | Média | ✅ Concluído |
| S4-07 | Salvar `classification_report.json` como artefato por modelo | Baixa | ✅ Concluído |

---

### Sprint 5 — Pipeline RAG ✅

| ID | Item | Prioridade | Status |
|---|---|---|---|
| S5-01 | Implementar `retriever.py` — busca vetorial com MMR reranking | Alta | ✅ Concluído |
| S5-02 | Implementar `generator.py` — montagem de prompt few-shot + chamada ao Ollama | Alta | ✅ Concluído |
| S5-03 | Implementar `pipeline.py` — orquestração do fluxo RAG completo | Alta | ✅ Concluído |
| S5-04 | Suporte a streaming de tokens (resposta em tempo real) | Alta | ✅ Concluído |
| S5-05 | Configurar threshold de similaridade mínima (`min_score`) | Média | ✅ Concluído |
| S5-06 | Expor fontes/contexto junto com a resposta | Média | ✅ Concluído |

---

### Sprint 6 — API REST ✅

| ID | Item | Prioridade | Status |
|---|---|---|---|
| S6-01 | Implementar `POST /chat` — resposta completa em JSON | Alta | ✅ Concluído |
| S6-02 | Implementar `GET /chat/stream` — resposta em streaming SSE | Alta | ✅ Concluído |
| S6-03 | Implementar `GET /health` — status de todos os serviços | Alta | ✅ Concluído |
| S6-04 | Criar schemas Pydantic para validação de entrada e saída | Média | ✅ Concluído |
| S6-05 | Documentação automática Swagger em `/docs` | Média | ✅ Concluído |
| S6-06 | Tratamento de erros e timeouts do LLM | Média | ✅ Concluído |

---

### Sprint 7 — Frontend (Gradio) ✅

| ID | Item | Prioridade | Status |
|---|---|---|---|
| S7-01 | Implementar `app.py` com interface de chat Gradio | Alta | ✅ Concluído |
| S7-02 | Integrar com API via streaming (`/chat/stream`) | Alta | ✅ Concluído |
| S7-03 | Toggle para exibir documentos usados como contexto | Média | ✅ Concluído |
| S7-04 | Exemplos de perguntas pré-definidos | Baixa | ✅ Concluído |

---

### Sprint 8 — Avaliação e Qualidade ✅

| ID | Item | Prioridade | Status |
|---|---|---|---|
| S8-01 | Implementar `eval_rag.py` — avaliação de retrieval (Precision@K, MRR) | Alta | ✅ Concluído |
| S8-02 | Modo `retrieval_only` para avaliação rápida sem LLM | Média | ✅ Concluído |
| S8-03 | Modo `full` para avaliação completa com LLM | Média | ✅ Concluído |
| S8-04 | Documentar resultados de avaliação | Baixa | ✅ Concluído |

---

### Sprint 9 — Documentação Final ✅

| ID | Item | Prioridade | Status |
|---|---|---|---|
| S9-01 | README.md completo com todas as seções | Alta | ✅ Concluído |
| S9-02 | Diagrama de arquitetura completo | Alta | ✅ Concluído |
| S9-03 | Backlog completo documentado | Alta | ✅ Concluído |
| S9-04 | Documento de governança de dados | Alta | ✅ Concluído |
| S9-05 | Guia de apresentação | Média | ✅ Concluído |
| S9-06 | CHECKLIST.md de entregáveis | Baixa | ✅ Concluído |

---

## 9. Governança de Dados

### 9.1 Princípios Gerais

O AutoPredict AI adota os seguintes princípios de governança:

- **Rastreabilidade** — todo dado possui origem documentada e é versionado por camada (Bronze/Silver/Gold)
- **Isolamento** — todos os recursos Docker usam o namespace `autopredict_`, sem interferência com outros projetos
- **Auditoria** — cada operação de ingestão é registrada com timestamp e contagem de registros no PostgreSQL
- **Reprodutibilidade** — todos os experimentos de ML são registrados no MLflow com parâmetros, métricas e artefatos

---

### 9.2 Padrão Medallion — Política de Dados por Camada

| Camada | Princípio | O que é armazenado | Localização | Permissão de escrita |
|---|---|---|---|---|
| **Bronze** | Dado bruto, imutável | CSV original sem modificação | MinIO `bronze/` | Apenas pipeline de ingestão |
| **Silver** | Dado limpo, padronizado | CSV sem nulos, colunas normalizadas | MinIO `silver/` | Apenas pipeline Silver |
| **Gold** | Dado pronto para IA | CSV final + vetores indexados | MinIO `gold/` + Milvus | Apenas pipeline Gold |

> **Regra:** Nenhum processo acessa a camada Gold diretamente para escrita. Apenas o pipeline Gold (`src/data_pipeline/gold.py`) insere documentos no Milvus.

---

### 9.3 Qualidade de Dados

Validações aplicadas na camada Silver:

| Verificação | Ação tomada |
|---|---|
| Valores nulos | Remoção da linha |
| Tipos de coluna incorretos | Conversão automática ou descarte |
| Duplicatas exatas | Remoção, mantendo primeira ocorrência |
| Datasets grandes (>5.000 linhas) | Limitado a 5.000 para indexação no Gold |

---

### 9.4 Versionamento de Modelos ML

Todos os modelos treinados são registrados no **MLflow Model Registry** com:

- **Nome padronizado:** `AutoPredict-{Experimento}__{algoritmo}` (ex: `AutoPredict-Maintenance__xgboost`)
- **Métricas obrigatórias:** Accuracy, F1-score (weighted), AUC-ROC
- **Artefatos obrigatórios:** `classification_report.json`, `model.pkl`
- **Parâmetros:** hiperparâmetros do modelo, dataset utilizado, data de treino

| Experimento | Modelos registrados | Melhor modelo |
|---|---|---|
| AutoPredict-Maintenance | logistic_regression, random_forest, xgboost | xgboost (F1=1.00) |
| AutoPredict-Predictive | logistic_regression, random_forest, xgboost | random_forest (F1=0.56) |
| AutoPredict-EngineFault | logistic_regression, random_forest, xgboost | registrados no MLflow |

---

### 9.5 Segurança e Credenciais

| Credencial | Padrão | Como alterar |
|---|---|---|
| MinIO usuário/senha | `minioadmin` / `minioadmin123` | Variável `MINIO_USER` / `MINIO_PASSWORD` no `.env` |
| PostgreSQL usuário/senha | `autopredict` / `autopredict123` | Variável `POSTGRES_USER` / `POSTGRES_PASSWORD` no `.env` |
| Ollama | sem autenticação (interno) | — |
| Milvus | sem autenticação (interno) | — |

> **Atenção:** As credenciais acima são para ambiente de desenvolvimento. Em produção, utilize um arquivo `.env` com valores seguros e **nunca** commite credenciais no repositório.

---

### 9.6 Isolamento entre Projetos Docker

O `docker-compose.yml` define `name: autopredict`, o que garante que todos os recursos (containers, volumes, redes) sejam criados com o prefixo `autopredict_`. Isso impede conflitos com outros projetos Docker no mesmo host.

| Recurso | Nome criado pelo Compose |
|---|---|
| Rede | `autopredict_autopredict-network` |
| Volume MinIO | `autopredict_minio_data` |
| Volume Milvus | `autopredict_milvus_data` |
| Volume PostgreSQL | `autopredict_postgres_data` |
| Volume MLflow | `autopredict_mlflow_data` |
| Volume Ollama | `autopredict_ollama_data` |

---

## 10. Comandos Úteis

### Re-indexar o Milvus (após trocar o modelo de embedding)

```bash
# Dropar e recriar coleções
docker exec -w /app autopredict-api python -c "
from src.database.milvus_client import MilvusClient
c = MilvusClient()
for col in ['vehicle_maintenance', 'car_predictive', 'engine_fault']:
    c.drop_collection(col)
    c.create_collection(col)
    print(col, '→ recriada')
"

# Re-indexar
docker exec -w /app autopredict-api python -m src.data_pipeline.gold
```

### Re-treinar modelos ML

```bash
docker exec -w /app autopredict-api python -m src.ml.train
```

### Rodar avaliação de métricas (retrieval + RAG)

```bash
# Apenas retrieval (rápido, ~1 min):
docker exec -w /app autopredict-api python -m src.evaluation.eval_rag retrieval_only

# Pipeline completo com LLM (lento, ~10 min):
docker exec -w /app autopredict-api python -m src.evaluation.eval_rag full 4
```

### Ver logs de um serviço

```bash
docker logs autopredict-api --tail 50
docker logs autopredict-frontend --tail 50
docker logs autopredict-ollama --tail 50
```

### Reconstruir a imagem da API (após alterar requirements.txt)

```bash
docker compose up -d --build api
```

---

## 11. Solução de Problemas

### "Não foi possível conectar à API" no frontend

Verifique se todos os containers estão rodando:

```bash
docker compose ps
```

Se algum estiver `Exited`, veja o motivo:

```bash
docker logs autopredict-api
```

### Respostas com "⚠️ O modelo demorou muito para responder"

O LLM ainda está sendo carregado ou a fila está ocupada. Aguarde 1-2 minutos e tente novamente. Isso é normal na primeira pergunta após subir os containers.

### Milvus com 0 documentos após subir os containers

O pipeline Gold precisa ser re-executado. Execute o Passo 2 da seção [Primeiros Passos](#4-primeiros-passos-setup-inicial).

### `FileNotFoundError` ao rodar o pipeline

A pasta `data/` não existe ou os CSVs não foram colocados lá. Veja a seção [Estrutura de pastas](#22-estrutura-de-pastas-necessária).

### Erro de porta em uso (`bind: address already in use`)

Alguma porta (7860, 8000, 5001, etc.) está sendo usada por outro processo. Identifique e pare o processo, ou altere a porta no `docker-compose.yml`.

### Erro no XGBoost (`num_class`)

Isso era um bug da versão anterior, já corrigido. Certifique-se de estar na versão mais recente:

```bash
git pull
docker compose up -d --build api
```

---

## Equipe

| RA | Nome |
|---|---|
| 190435 | Matheus Diogo Teixeira |
| 200817 | Adrian Antonio de Oliveira |
| 212199 | Eduardo Piratello |
| 222239 | Giovana Antunes Soares |
| 222312 | Heloisa Goulart Vicencio |
| 212146 | Juliane Zaetum de Oliveira |
| 212109 | Larissa Cezar Eringer |
| 222236 | Lucas Martins |
| 222804 | Nicolas Andrade De Marchi Nicolau |
| 222255 | Pedro Henrique Cavalheiro Modesto |
