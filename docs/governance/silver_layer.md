# Governança de Dados — Camada Silver

## 📋 Sumário Executivo

A **camada Silver** implementa a **limpeza, normalização e padronização** dos dados brutos provenientes da camada Bronze, preparando-os para análise e modelagem de Machine Learning.

---

## 🎯 Propósito e Responsabilidades

### Definição
A camada Silver é responsável por transformar dados **brutos e inconsistentes** em dados **limpos, validados e estruturados**, mantendo a semântica original sem perda de informação crítica.

### Princípios
- ✅ **Qualidade de Dados** — Aplicação de validações e limpezas obrigatórias
- ✅ **Consistência** — Padronização de formatos e tipos de dados
- ✅ **Integridade Referencial** — Remoção de valores impossíveis ou contraditórios
- ✅ **Rastreabilidade** — Registro de quantas linhas foram descartadas e por quê

---

## 📁 Estrutura de Armazenamento

### Localização Física
- **Sistema:** MinIO (S3-compatible object storage)
- **Bucket:** `silver`
- **Formato:** CSV limpo e normalizado

### Convenção de Nomenclatura
```
silver/
├── silver_vehicle_maintenance_data.csv       # Bronze → limpo
├── silver_cars_hyundai.csv                   # Bronze → limpo
└── silver_engine_fault_detection_dataset.csv # Bronze → limpo
```

> **Padrão:** `silver_{nome_arquivo_bronze}.csv`

---

## 🔄 Transformações Aplicadas

### 1. Limpeza de Valores Nulos
```python
# Regra: Remover linhas com qualquer valor nulo
df = df.dropna()
```

**Justificativa:** Modelos de ML e embeddings não aceitam valores ausentes. Imputação foi considerada mas rejeitada para preservar integridade.

| Dataset | Linhas Bronze | Linhas Silver | % Perdido |
|---|---|---|---|
| vehicle_maintenance | ~50.000 | ~50.000 | 0% |
| cars_hyundai | 1.100 | 1.100 | 0% |
| engine_fault | 10.000 | 10.000 | 0% |

### 2. Normalização de Nomes de Colunas

**Problema:** Colunas com caracteres especiais causam erros em pipelines:
- `Temperature(°C)` → erro de encoding
- `Vibration (mm)` → espaços dificultam acesso

**Solução:** Regex que remove caracteres especiais:
```python
df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True).str.strip().str.replace(' ', '_')
```

**Exemplo de Transformações:**
| Bronze | Silver |
|---|---|
| `Temperature(°C)` | `Temperature` |
| `Vibration (mm)` | `Vibration` |
| `Pressure(PSI)` | `Pressure` |
| `Humidity(%)` | `Humidity` |
| `Engine Hours` | `Engine_Hours` |

### 3. Conversão e Validação de Tipos

| Coluna | Tipo Original | Tipo Silver | Validação |
|---|---|---|---|
| `Mileage` | string | float | Valores negativos → descartados |
| `Last_Service_Date` | string | datetime | Formato ISO 8601 |
| `Oil_Quality` | int | category | Valores válidos: 1-5 |
| `Need_Maintenance` | int | bool | 0 ou 1 apenas |
| `Anomaly_Indication` | int | bool | 0 ou 1 apenas |
| `Engine_Condition` | int | int | 0, 1 ou 2 apenas |

### 4. Detecção de Target Variables

O pipeline Silver identifica automaticamente as colunas-alvo para Machine Learning:

```python
# Dataset 1: vehicle_maintenance
if 'Need_Maintenance' in df.columns or 'need_maintenance' in df.columns:
    target_col = 'need_maintenance'
    
# Dataset 2: cars_hyundai
if 'Anomaly_Indication' in df.columns or 'anomaly_indication' in df.columns:
    target_col = 'anomaly_indication'
    
# Dataset 3: engine_fault
if 'Engine_Condition' in df.columns or 'engine_condition' in df.columns:
    target_col = 'engine_condition'
```

### 5. Remoção de Duplicatas

```python
# Remove linhas idênticas, mantendo a primeira ocorrência
df = df.drop_duplicates()
```

---

## 🔐 Políticas de Acesso e Segurança

### Permissões de Escrita
| Componente | Permissão | Justificativa |
|---|---|---|
| `src/data_pipeline/silver.py` | ✅ Escrita (sobrescreve) | Único responsável pela transformação |
| Pipeline Gold | 🔍 Somente leitura | Consome dados limpos |
| Pipeline Bronze | ❌ Sem acesso | Fluxo unidirecional Bronze→Silver |
| API/Frontend | ❌ Sem acesso direto | Isolamento de camadas |
| Desenvolvedores | 🔍 Leitura via MinIO Console | Validação e debugging |

---

## ⚙️ Processo de Transformação

### Fluxo Automatizado
```
┌─────────────────────┐
│  MinIO Bronze       │
│  (CSVs brutos)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  silver.py          │
│  process_to_silver()│
└──────────┬──────────┘
           │
           ├──► Leitura do Bronze
           ├──► Dropna()
           ├──► Normaliza colunas
           ├──► Valida tipos
           ├──► Drop duplicates
           │
           ▼
    MinIO.upload_file()
           │
           ▼
    Bucket silver/
           │
           └──► PostgresClient.log_ingestion()
                      │
                      ▼
                Tabela: ingestion_log
```

### Script de Execução
```bash
docker exec -w /app autopredict-api python -c "
from src.data_pipeline.silver import process_to_silver
result = process_to_silver()
print(result)
"
```

### Output Esperado
```json
{
  "status": "success",
  "files_processed": 3,
  "silver_bucket": "silver",
  "files": [
    "silver_vehicle_maintenance_data.csv",
    "silver_cars_hyundai.csv",
    "silver_engine_fault_detection_dataset.csv"
  ]
}
```

---

## 📈 Métricas de Qualidade (Silver)

### KPIs Monitorados
| Métrica | Descrição | Meta |
|---|---|---|
| **Taxa de Completude** | % linhas sem nulos após limpeza | 100% |
| **Taxa de Transformação** | % linhas Bronze → Silver | ≥ 95% |
| **Duplicatas Removidas** | Qtd de linhas duplicadas | < 1% |
| **Colunas Normalizadas** | % colunas com nomes padronizados | 100% |
| **Tipos Validados** | % colunas com tipo correto | 100% |

### Validações Aplicadas na Silver
✅ **Nulos removidos** — `df.dropna()`  
✅ **Duplicatas removidas** — `df.drop_duplicates()`  
✅ **Colunas normalizadas** — Regex para remover caracteres especiais  
✅ **Target identificada** — Detecção automática de coluna-alvo para ML  
❌ **Outliers preservados** — Tratamento de outliers é responsabilidade da camada Gold (feature engineering)

---

## 📊 Schema de Dados Silver

### Dataset 1: Vehicle Maintenance (Silver)
```
Colunas: 15
Linhas: ~50.000
Target: need_maintenance (bool)

Principais colunas:
- vehicle_id (int)
- mileage (float)
- last_service_date (datetime)
- engine_hours (float)
- oil_quality (int, 1-5)
- brake_condition (str, "Good"/"Fair"/"Poor")
- tire_pressure (float, PSI)
- need_maintenance (bool, target)
```

### Dataset 2: Cars Hyundai (Silver)
```
Colunas: 8
Linhas: ~1.100
Target: anomaly_indication (bool)

Principais colunas:
- temperature (float, Celsius)
- vibration (float, mm)
- pressure (float, PSI)
- humidity (float, %)
- anomaly_indication (bool, target)
```

### Dataset 3: Engine Fault Detection (Silver)
```
Colunas: 5
Linhas: ~10.000
Target: engine_condition (int, multiclass)

Principais colunas:
- vibration_1 (float)
- vibration_2 (float)
- temperature (float, Celsius)
- rpm (int)
- engine_condition (int: 0=normal, 1=alerta, 2=crítico)
```

---

## 📝 Rastreabilidade e Auditoria

### Logs de Transformação
Cada execução do pipeline Silver registra:
- Timestamp do processamento
- Quantidade de linhas de entrada (Bronze)
- Quantidade de linhas de saída (Silver)
- Taxa de descarte (%)

### Consulta SQL — Histórico Silver
```sql
SELECT 
    source_file,
    records_count,
    timestamp,
    destination
FROM ingestion_log
WHERE layer = 'silver'
ORDER BY timestamp DESC;
```

---

## 🚨 Troubleshooting

### Erro: "MinIO bronze bucket empty"
**Causa:** Pipeline Bronze não foi executado  
**Solução:** Rodar `bronze.py` antes de `silver.py`

### Erro: "Target column not found"
**Causa:** Nome da coluna-alvo mudou no dataset original  
**Solução:** Atualizar detecção de target em `silver.py` (linhas 60-75)

### Aviso: "50% of rows dropped due to nulls"
**Causa:** Dataset Bronze tem muitos valores ausentes  
**Solução:** Revisar fonte de dados original. Se esperado, documentar perda de dados.

---

## 🔄 Reprocessamento

Para reprocessar a camada Silver:

```bash
# Re-executar pipeline Silver
docker exec -w /app autopredict-api python -c "
from src.data_pipeline.silver import process_to_silver
process_to_silver()
"
```

> ⚠️ **Importante:** Isso sobrescreverá os arquivos existentes no bucket `silver/`.

---

## 📋 Checklist de Validação Silver

Após executar o pipeline, verificar:

- [ ] 3 arquivos criados no bucket `silver/`
- [ ] Nomes de arquivos seguem padrão `silver_{nome}.csv`
- [ ] Nenhuma linha com valores nulos (validar com `df.isnull().sum()`)
- [ ] Colunas sem caracteres especiais (validar nomes de colunas)
- [ ] Target identificada corretamente em cada dataset
- [ ] Registro no PostgreSQL `ingestion_log` com `layer='silver'`

---

## 📅 Histórico de Mudanças

| Data | Versão | Mudança | Responsável |
|---|---|---|---|
| 2026-05-14 | 1.0 | Criação inicial do documento | AutoPredict Team |

---

## 📚 Referências

- [Pandas Data Cleaning Best Practices](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Medallion Architecture — Silver Layer](https://www.databricks.com/glossary/medallion-architecture#silver-layer)
- [Data Quality Metrics for ML](https://neptune.ai/blog/data-quality-metrics)

---

**Aprovação:** Product Owner AutoPredict AI  
**Revisão:** Scrum Master AutoPredict AI  
**Próxima revisão:** Sprint 10
