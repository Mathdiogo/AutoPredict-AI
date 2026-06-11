# Governança de Dados — Camada Gold

## 📋 Sumário Executivo

A **camada Gold** é a camada de **produção** que contém dados otimizados, enriquecidos e prontos para consumo por aplicações de IA, modelos de Machine Learning e sistemas de busca semântica (RAG).

---

## 🎯 Propósito e Responsabilidades

### Definição
A camada Gold transforma dados limpos (Silver) em **artefatos de alto valor**, incluindo embeddings vetoriais, chunks de texto otimizados e metadados enriquecidos para recuperação eficiente via RAG (Retrieval-Augmented Generation).

### Princípios
- ✅ **Otimização para IA** — Dados estruturados especificamente para modelos de linguagem
- ✅ **Busca Semântica** — Vetores indexados no Milvus para similaridade cosine
- ✅ **Performance** — Chunking otimizado para contexto de LLM (limite de tokens)
- ✅ **Governança de Vetores** — Rastreabilidade entre texto → embedding → documento Milvus

---

## 📁 Estrutura de Armazenamento

### Dupla Persistência
A camada Gold armazena dados em **dois sistemas complementares**:

#### 1. MinIO (arquivos CSV finais)
```
gold/
├── gold_vehicle_maintenance_data.csv       # Chunks textuais + metadados
├── gold_cars_hyundai.csv                   # Chunks textuais + metadados
└── gold_engine_fault_detection_dataset.csv # Chunks textuais + metadados
```

#### 2. Milvus (banco vetorial)
```
Coleções Milvus:
├── vehicle_maintenance     # 5.000 documentos indexados
├── car_predictive          # 1.100 documentos indexados
└── engine_fault            # 5.000 documentos indexados

Total: 11.100 documentos vetorizados
```

---

## 🔄 Transformações Aplicadas

### 1. Chunking Estratégico

Cada linha do dataset Silver é transformada em um **chunk de texto** descritivo para o LLM entender.

#### Exemplo — Dataset Vehicle Maintenance
**Linha Silver (CSV):**
```csv
vehicle_id,mileage,oil_quality,brake_condition,need_maintenance
V-001,85000,2,Fair,1
```

**Chunk Gold (texto):**
```
Veículo V-001 com 85000 km rodados. Qualidade do óleo: 2/5 (baixa). 
Condição dos freios: Fair. Status de manutenção: necessita manutenção preventiva.
```

**Razão:** LLMs entendem texto natural, não tabelas CSV.

#### Exemplo — Dataset Cars Hyundai
**Linha Silver:**
```csv
temperature,vibration,pressure,humidity,anomaly_indication
92.5,3.2,34.1,68.0,1
```

**Chunk Gold:**
```
Sensores: temperatura 92.5°C, vibração 3.2 mm, pressão 34.1 PSI, umidade 68.0%. 
Indicador de anomalia: detectado problema.
```

### 2. Geração de Embeddings

```python
# Modelo: paraphrase-multilingual-MiniLM-L12-v2
# Dimensões: 384
# Treinado em 50+ idiomas incluindo português

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embedding = model.encode(chunk_text)  # → array de 384 floats
```

**Características do modelo:**
- ✅ Multilíngue (português otimizado)
- ✅ 384 dimensões (rápido e eficiente)
- ✅ Normalizado para similaridade cosine
- ✅ ~120MB (download na primeira execução)

### 3. Indexação no Milvus

Cada documento inserido no Milvus contém:

```python
document = {
    "text": str,          # Chunk de texto descritivo
    "source": str,        # Nome do dataset original
    "metadata": dict,     # JSON com colunas-chave do registro
    "embedding": list     # Vetor de 384 dimensões
}
```

#### Schema Milvus
```python
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="metadata", dtype=DataType.JSON),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
]
```

#### Índice de Busca
```python
index_params = {
    "metric_type": "COSINE",    # Similaridade cosine
    "index_type": "IVF_FLAT",   # Inverted File index
    "params": {"nlist": 128}    # 128 clusters
}
```

**Justificativa do índice:**
- `COSINE` → Embeddings normalizados funcionam melhor com cosine
- `IVF_FLAT` → Balanceamento entre velocidade e precisão para 11k docs
- `nlist=128` → Suficiente para dataset de médio porte

### 4. Limitação de Tamanho (Subsampling)

```python
# Regra: Máximo de 5.000 documentos por coleção
if len(df) > 5000:
    df = df.sample(n=5000, random_state=42)
```

**Razão:**
- Performance de busca no Milvus
- Tempo de indexação aceitável (<2 min por dataset)
- Limite de contexto do LLM (top_k=5 docs × 3 datasets = 15 docs no prompt)

---

## 🔐 Políticas de Acesso e Segurança

### Permissões de Escrita
| Componente | Permissão | Justificativa |
|---|---|---|
| `src/data_pipeline/gold.py` | ✅ Escrita completa | Único responsável pela indexação |
| `src/rag/retriever.py` | 🔍 Somente leitura (busca vetorial) | Consumidor principal |
| Pipelines Bronze/Silver | ❌ Sem acesso | Fluxo unidirecional |
| API/Frontend | ❌ Sem acesso direto | Acesso via `retriever.py` |

### Credenciais Milvus
```
Host: localhost
Port: 19530 (gRPC)
Auth: Sem autenticação (padrão standalone)
```

> ⚠️ **PRODUÇÃO:** Habilitar autenticação e TLS no Milvus.

---

## ⚙️ Processo de Geração

### Fluxo Automatizado
```
┌─────────────────────┐
│  MinIO Silver       │
│  (CSVs limpos)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  gold.py            │
│  process_to_gold()  │
└──────────┬──────────┘
           │
           ├──► Leitura do Silver
           ├──► Chunking (row → texto descritivo)
           ├──► Embedding (SentenceTransformer)
           ├──► Subsampling (max 5k por dataset)
           │
           ▼
    MilvusClient.insert()
           │
           ├──► Coleção: vehicle_maintenance
           ├──► Coleção: car_predictive
           └──► Coleção: engine_fault
           │
           ▼
    MinIO.upload_file()
           │
           ▼
    Bucket gold/ (CSV com chunks)
           │
           └──► PostgresClient.log_ingestion()
```

### Script de Execução
```bash
# Gerar embeddings e indexar no Milvus (full pipeline)
docker exec -w /app autopredict-api python -m src.data_pipeline.run_pipeline
```

Ou executar apenas Gold:
```bash
docker exec -w /app autopredict-api python -c "
from src.data_pipeline.gold import process_to_gold
result = process_to_gold()
print(result)
"
```

### Output Esperado
```json
{
  "status": "success",
  "collections_created": 3,
  "total_documents_indexed": 11100,
  "collections": {
    "vehicle_maintenance": 5000,
    "car_predictive": 1100,
    "engine_fault": 5000
  },
  "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
  "embedding_dim": 384
}
```

---

## 📈 Métricas de Qualidade (Gold)

### KPIs Monitorados
| Métrica | Descrição | Meta |
|---|---|---|
| **Taxa de Indexação** | % docs Silver → Milvus | 100% (ou sample 5k) |
| **Dimensionalidade** | Vetores com 384 dims | 100% |
| **Latência de Busca** | Tempo de retrieval top-5 | < 100ms |
| **Precisão@5** | Docs relevantes nos top-5 | ≥ 80% |
| **Normalização de Vetores** | ||v|| = 1.0 (cosine) | 100% |

### Validações Aplicadas na Gold
✅ **Embeddings normalizados** — Verificação de magnitude do vetor  
✅ **Chunks não-vazios** — Descarte de linhas sem texto descritivo  
✅ **Schema Milvus validado** — 5 campos obrigatórios presentes  
✅ **Index construído** — Índice `IVF_FLAT` criado após inserção  

---

## 🔍 Uso em Produção (RAG Pipeline)

### Fluxo de Busca Semântica
```
Usuário: "Quais os sinais de que o freio precisa de manutenção?"
           │
           ▼
    Embedder.encode(pergunta)  → vetor_query [384 dims]
           │
           ▼
    Milvus.search(vetor_query, top_k=5)
           │
           ├──► Coleção: vehicle_maintenance (top 5)
           ├──► Coleção: car_predictive (top 5)
           └──► Coleção: engine_fault (top 5)
           │
           ▼
    Total: 15 documentos relevantes
           │
           ▼
    Contexto montado para LLM (Ollama)
           │
           ▼
    Resposta fundamentada com fontes citadas
```

### Parâmetros de Busca
```python
search_params = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10},  # Quantos clusters IVF buscar
    "top_k": 5                 # Top 5 docs por coleção
}
```

---

## 📊 Estatísticas dos Documentos Indexados

### Distribuição por Coleção
| Coleção | Documentos | Origem Silver | % Utilizado |
|---|---|---|---|
| vehicle_maintenance | 5.000 | 50.000 | 10% (sample) |
| car_predictive | 1.100 | 1.100 | 100% |
| engine_fault | 5.000 | 10.000 | 50% (sample) |
| **TOTAL** | **11.100** | **61.100** | **18%** |

### Características dos Chunks
| Dataset | Tamanho médio do chunk | Tokens médios | Range de tokens |
|---|---|---|---|
| vehicle_maintenance | ~180 chars | ~45 tokens | 30-80 |
| car_predictive | ~150 chars | ~38 tokens | 25-60 |
| engine_fault | ~120 chars | ~30 tokens | 20-50 |

> **Contexto LLM:** Com top_k=5 e 3 datasets, o contexto total fica em ~1.500 tokens, deixando espaço para o prompt e resposta (limite do llama3.2:3b = 8K tokens).

---

## 📝 Rastreabilidade e Auditoria

### Metadata Armazenada por Documento
Cada documento no Milvus carrega metadados ricos:

```json
{
  "id": 12345,
  "text": "Veículo V-001 com 85000 km...",
  "source": "vehicle_maintenance_data",
  "metadata": {
    "vehicle_id": "V-001",
    "mileage": 85000,
    "brake_condition": "Fair",
    "need_maintenance": true,
    "indexed_at": "2026-05-14T10:30:00Z"
  },
  "embedding": [0.023, -0.145, ..., 0.089]  # 384 floats
}
```

### Consulta SQL — Logs de Indexação Gold
```sql
SELECT 
    source_file,
    destination,
    records_count,
    timestamp
FROM ingestion_log
WHERE layer = 'gold'
ORDER BY timestamp DESC;
```

### Verificar Contagem de Documentos
```python
from src.database.milvus_client import MilvusClient

client = MilvusClient()
for collection in ['vehicle_maintenance', 'car_predictive', 'engine_fault']:
    count = client.get_count(collection)
    print(f"{collection}: {count} docs")
```

---

## 🚨 Troubleshooting

### Erro: "Milvus connection refused"
**Causa:** Container `autopredict-milvus` não está rodando  
**Solução:** `docker compose up -d milvus` e aguardar health check

### Erro: "Collection already exists"
**Causa:** Pipeline Gold foi executado anteriormente  
**Solução:** Dropar coleções e recriar:
```python
from src.database.milvus_client import MilvusClient
c = MilvusClient()
c.drop_collection('vehicle_maintenance')
c.drop_collection('car_predictive')
c.drop_collection('engine_fault')
```

### Aviso: "Embedding model downloading (~120MB)"
**Causa:** Primeira execução do pipeline Gold  
**Solução:** Aguardar download do modelo (1-2 min). Downloads futuros usam cache.

### Erro: "Index build failed — too few documents"
**Causa:** Menos de 100 documentos na coleção  
**Solução:** Ajustar `nlist` no `create_index()` ou aumentar tamanho do dataset

---

## 🔄 Reprocessamento

### Re-indexar Todas as Coleções
```bash
# 1. Dropar coleções existentes
docker exec -w /app autopredict-api python -c "
from src.database.milvus_client import MilvusClient
c = MilvusClient()
for col in ['vehicle_maintenance', 'car_predictive', 'engine_fault']:
    c.drop_collection(col)
"

# 2. Re-executar pipeline Gold
docker exec -w /app autopredict-api python -m src.data_pipeline.gold
```

> ⚠️ **Downtime:** Durante a re-indexação, a API retornará erro em queries. Planeje janela de manutenção.

---

## 🎯 Otimizações Futuras

| Otimização | Impacto | Complexidade |
|---|---|---|
| **HNSW index** (em vez de IVF_FLAT) | 🚀 Busca 10x mais rápida | Baixa |
| **Reranking com Cross-Encoder** | 📈 +15% precisão | Média |
| **Dynamic chunking** (overlap 10%) | 📖 Contexto mais rico | Média |
| **Incremental indexing** | 🔄 Adicionar docs sem rebuild | Alta |
| **GPU acceleration** | ⚡ Embedding 5x mais rápido | Alta (requer CUDA) |

---

## 📋 Checklist de Validação Gold

Após executar o pipeline, verificar:

- [ ] 3 coleções criadas no Milvus
- [ ] Total de 11.100 documentos indexados
- [ ] Índice `IVF_FLAT` construído em cada coleção
- [ ] 3 arquivos CSV no bucket `gold/` do MinIO
- [ ] Query de teste retorna resultados relevantes
- [ ] Logs no PostgreSQL `ingestion_log` com `layer='gold'`

---

## 📅 Histórico de Mudanças

| Data | Versão | Mudança | Responsável |
|---|---|---|---|
| 2026-05-14 | 1.0 | Criação inicial do documento | AutoPredict Team |

---

## 📚 Referências

- [Milvus Vector Database Documentation](https://milvus.io/docs)
- [Sentence-Transformers Library](https://www.sbert.net/)
- [RAG Best Practices — Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [Medallion Architecture — Gold Layer](https://www.databricks.com/glossary/medallion-architecture#gold-layer)

---

**Aprovação:** Product Owner AutoPredict AI  
**Revisão:** Scrum Master AutoPredict AI  
**Próxima revisão:** Sprint 10
