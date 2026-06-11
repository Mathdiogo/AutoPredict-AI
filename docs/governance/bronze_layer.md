# Governança de Dados — Camada Bronze

## 📋 Sumário Executivo

A **camada Bronze** é a porta de entrada de todos os dados brutos no sistema AutoPredict AI. Ela implementa o princípio de **imutabilidade** e **rastreabilidade total** da origem dos dados.

---

## 🎯 Propósito e Responsabilidades

### Definição
A camada Bronze armazena os **dados brutos originais** sem nenhuma transformação, limpeza ou validação. É a camada de **ingestão pura**.

### Princípios
- ✅ **Imutabilidade** — Os dados nunca são alterados após a ingestão
- ✅ **Auditoria completa** — Timestamp e origem de cada arquivo registrados no PostgreSQL
- ✅ **Single Source of Truth** — Única fonte oficial dos dados brutos
- ✅ **Reprodutibilidade** — Permite reprocessar Silver/Gold a qualquer momento

---

## 📁 Estrutura de Armazenamento

### Localização Física
- **Sistema:** MinIO (S3-compatible object storage)
- **Bucket:** `bronze`
- **Formato:** CSV original sem modificações

### Convenção de Nomenclatura
```
bronze/
├── vehicle_maintenance_data.csv       # Dataset 1 — 50.000 registros
├── cars_hyundai.csv                   # Dataset 2 — 1.100 registros
└── engine_fault_detection_dataset.csv # Dataset 3 — 10.000 registros
```

> **Nota:** Os nomes dos arquivos são preservados exatamente como estão nos datasets originais do Kaggle.

---

## 📊 Datasets Catalogados

### Dataset 1: Vehicle Maintenance Data
- **Origem:** [Kaggle - Vehicle Maintenance Data](https://www.kaggle.com/datasets/chavindudulaj/vehicle-maintenance-data)
- **Descrição:** Histórico de manutenção preventiva e corretiva de frotas comerciais
- **Registros:** ~50.000 linhas
- **Colunas principais:** 
  - `Vehicle_ID`, `Mileage`, `Last_Service_Date`, `Engine_Hours`
  - `Oil_Quality`, `Brake_Condition`, `Tire_Pressure`
  - `Need_Maintenance` (target para ML)

### Dataset 2: Cars Hyundai (Predictive Maintenance)
- **Origem:** [Kaggle - Car Predictive Maintenance](https://www.kaggle.com/datasets/pragyanaianddsschool/car-predictive-maintenance-data)
- **Descrição:** Dados de sensores e indicadores de anomalias em veículos Hyundai
- **Registros:** ~1.100 linhas
- **Colunas principais:**
  - Sensores: `Temperature(°C)`, `Vibration(mm)`, `Pressure(PSI)`, `Humidity(%)`
  - `Anomaly_Indication` (target para ML)

### Dataset 3: Engine Fault Detection
- **Origem:** [Kaggle - Engine Fault Detection Data](https://www.kaggle.com/datasets/ziya07/engine-fault-detection-data)
- **Descrição:** Dados de vibração, temperatura e RPM para detecção de falhas no motor
- **Registros:** ~10.000 linhas
- **Colunas principais:**
  - `Vibration_1`, `Vibration_2`, `Temperature`, `RPM`
  - `Engine_Condition` (0=normal, 1=alerta, 2=crítico) — target multiclass

---

## 🔐 Políticas de Acesso e Segurança

### Permissões de Escrita
| Componente | Permissão | Justificativa |
|---|---|---|
| `src/data_pipeline/bronze.py` | ✅ Escrita (apenas inserção) | Único responsável pela ingestão |
| Pipelines Silver/Gold | ❌ Somente leitura | Não podem modificar origem |
| API/Frontend | ❌ Sem acesso direto | Isolamento de responsabilidades |
| Desenvolvedores | 🔍 Leitura via MinIO Console | Auditoria e troubleshooting |

### Credenciais MinIO (ambiente dev)
```
Endpoint: localhost:9000
Console:  http://localhost:9001
User:     minioadmin
Password: minioadmin123
```

> ⚠️ **PRODUÇÃO:** Utilizar credenciais seguras via `.env` e IAM policies do MinIO.

---

## ⚙️ Processo de Ingestão

### Fluxo Automatizado
```
┌─────────────────────┐
│  Datasets Locais    │
│  (pasta data/)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  bronze.py          │
│  ingest_to_bronze() │
└──────────┬──────────┘
           │
           ├──► MinIO.upload_file()
           │         │
           │         ▼
           │    Bucket bronze/
           │
           └──► PostgresClient.log_ingestion()
                      │
                      ▼
                Tabela: ingestion_log
```

### Script de Execução
```bash
docker exec -w /app autopredict-api python -c "
from src.data_pipeline.bronze import ingest_to_bronze
result = ingest_to_bronze('/app/data')
print(result)
"
```

### Output Esperado
```json
{
  "status": "success",
  "files_ingested": 3,
  "bronze_bucket": "bronze",
  "files": [
    "vehicle_maintenance_data.csv",
    "cars_hyundai.csv",
    "engine_fault_detection_dataset.csv"
  ]
}
```

---

## 📈 Métricas de Qualidade (Bronze)

### KPIs Monitorados
| Métrica | Descrição | Meta |
|---|---|---|
| **Completude de Ingestão** | % de arquivos esperados vs. carregados | 100% |
| **Integridade de Arquivo** | Checksum MD5 validado | Sim |
| **Latência de Ingestão** | Tempo para carregar todos os CSVs | < 30 segundos |
| **Registro de Auditoria** | Logs salvos no PostgreSQL | 100% |

### Validações Aplicadas na Bronze
❌ **Nenhuma validação de dados** — A camada Bronze aceita qualquer conteúdo  
✅ **Validação de existência do arquivo** — `FileNotFoundError` se CSV não existe  
✅ **Validação de conexão MinIO** — Retry automático 3x se falhar upload

---

## 📝 Rastreabilidade e Auditoria

### Tabela PostgreSQL: `ingestion_log`
```sql
CREATE TABLE ingestion_log (
    id SERIAL PRIMARY KEY,
    layer VARCHAR(10) NOT NULL,              -- 'bronze', 'silver', 'gold'
    source_file VARCHAR(255),                -- nome do arquivo original
    destination VARCHAR(255),                -- caminho no MinIO ou Milvus
    records_count INTEGER,                   -- quantidade de linhas
    timestamp TIMESTAMP DEFAULT NOW()        -- quando foi processado
);
```

### Consulta de Auditoria — Última Ingestão Bronze
```sql
SELECT layer, source_file, records_count, timestamp
FROM ingestion_log
WHERE layer = 'bronze'
ORDER BY timestamp DESC
LIMIT 10;
```

---

## 🚨 Troubleshooting

### Erro: "FileNotFoundError: data/vehicle_maintenance_data.csv"
**Causa:** Arquivos CSV não estão na pasta `data/`  
**Solução:** Baixar os datasets do Kaggle e colocar em `data/`

### Erro: "MinIO connection refused"
**Causa:** Container `autopredict-minio` não está rodando  
**Solução:** `docker compose up -d minio` e aguardar health check

### Erro: "Bucket 'bronze' does not exist"
**Causa:** Bucket não foi criado no MinIO  
**Solução:** Criar manualmente via Console MinIO (http://localhost:9001)

---

## 🔄 Reprocessamento (Re-ingestão)

Caso seja necessário reingerir os dados Bronze:

```bash
# 1. Limpar o bucket bronze (opcional)
# No console MinIO, deletar todos os arquivos do bucket 'bronze'

# 2. Re-executar ingestão
docker exec -w /app autopredict-api python -c "
from src.data_pipeline.bronze import ingest_to_bronze
ingest_to_bronze('/app/data')
"
```

> ⚠️ **Importante:** Isso não afeta Silver/Gold. Para reprocessar tudo, rode o pipeline completo.

---

## 📅 Histórico de Mudanças

| Data | Versão | Mudança | Responsável |
|---|---|---|---|
| 2026-05-14 | 1.0 | Criação inicial do documento | AutoPredict Team |

---

## 📚 Referências

- [MinIO Object Storage Documentation](https://min.io/docs/minio/linux/index.html)
- [Medallion Architecture (Databricks)](https://www.databricks.com/glossary/medallion-architecture)
- [Kaggle - Vehicle Maintenance Data](https://www.kaggle.com/datasets/chavindudulaj/vehicle-maintenance-data)

---

**Aprovação:** Product Owner AutoPredict AI  
**Revisão:** Scrum Master AutoPredict AI  
**Próxima revisão:** Sprint 10
