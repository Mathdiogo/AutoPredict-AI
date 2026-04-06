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
7. [Arquitetura](#7-arquitetura)
8. [Comandos Úteis](#8-comandos-úteis)
9. [Solução de Problemas](#9-solução-de-problemas)

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

## 7. Arquitetura

```
┌─────────────────────────────────────────────────┐
│        Interface Web (Gradio :7860)             │
└────────────────┬────────────────────────────────┘
                 │ HTTP
┌────────────────▼────────────────────────────────┐
│         API REST (FastAPI :8000)                │
│   POST /chat  |  GET /chat/stream  |  /health   │
└────────────────┬────────────────────────────────┘
                 │
     ┌───────────┴───────────┐
     │                       │
┌────▼────────────┐   ┌──────▼──────────────────┐
│  Retriever      │   │  Generator              │
│  ─────────      │   │  ─────────              │
│  Embedding →    │   │  Few-shot prompt +      │
│  Milvus search  │   │  Ollama (llama3.2:3b)   │
│  MMR reranking  │   │  Streaming response     │
└────────────────┘   └─────────────────────────┘
        │
┌───────▼─────────────────────────────────────┐
│  Milvus (banco vetorial)                    │
│  ├── vehicle_maintenance  (5.000 docs)      │
│  ├── car_predictive       (1.100 docs)      │
│  └── engine_fault         (5.000 docs)      │
└─────────────────────────────────────────────┘

Pipeline de Dados (Medallion):
  data/ (CSV) → Bronze (MinIO) → Silver (MinIO) → Gold (Milvus)

Modelos ML (MLflow):
  3 datasets × 3 algoritmos = 9 modelos registrados
  (LogisticRegression, RandomForest, XGBoost)
```

---

## 8. Comandos Úteis

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

## 9. Solução de Problemas

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
