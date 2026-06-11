# AutoPredict AI - Informações do Sistema

## Quem é AutoPredict AI?

AutoPredict AI é um sistema inteligente de manutenção preditiva automotiva que combina:

- **RAG (Retrieval-Augmented Generation)** com 3 datasets especializados
- **Machine Learning** para prever falhas antes que ocorram  
- **Múltiplos modelos LLM** (Ollama local, OpenAI, Anthropic, Groq)
- **Governança completa** com auditoria de tokens e métricas de inferência

### Missão

Prevenir falhas veiculares através de análise preditiva baseada em dados reais, reduzindo custos de manutenção e aumentando a segurança dos veículos.

## Modelos de Machine Learning Treinados

O sistema treina 3 algoritmos diferentes para cada um dos 3 datasets automotivos:

### 1. Logistic Regression (Baseline)

**Tipo:** Modelo linear de classificação binária

**Uso:** Baseline para comparação de desempenho

**Métricas Típicas:**
- Acurácia: 85-90%
- Treinamento rápido e interpretável

**Aplicação:** Predição de necessidade de manutenção e detecção de anomalias

### 2. Random Forest Classifier

**Tipo:** Ensemble de árvores de decisão

**Métricas Típicas:**
- Acurácia: 95-98%
- Precisão: 0.94-0.98
- Recall: 0.93-0.97
- F1-Score: 0.94-0.98

**Features Utilizadas:**
- Sensores: temperatura, vibração, pressão de óleo, RPM, carga
- Manutenção: quilometragem, tipo de serviço, dias desde última manutenção
- Falhas: códigos de erro, temperatura de escape, pressão de admissão

**Aplicação:** Predição de falhas de motor e condição de componentes

### 3. XGBoost (Gradient Boosting)

**Tipo:** Gradient boosting otimizado

**Métricas Típicas:**
- Acurácia: 96-99%
- Melhor desempenho geral na maioria dos casos
- Suporta classificação multiclasse

**Features:** Mesmas do Random Forest, com feature importance detalhada

**Aplicação:** Classificação de condições de motor (normal/atenção/crítico)

### Datasets e Experimentos

**3 Datasets Principais:**
1. **Vehicle Maintenance** (~10.000 registros) - Histórico de manutenções
2. **Predictive Sensors** (~50.000 registros) - Dados de sensores automotivos
3. **Engine Fault** (~5.000 registros) - Códigos de falha e diagnósticos

**Rastreamento MLflow:**
- 3 experimentos (um por dataset)
- 3 runs por experimento (um por algoritmo)
- Versionamento completo de modelos, parâmetros e métricas
- Matriz de confusão e curvas ROC registradas como artefatos

## Estratégias de Governança

### 1. Arquitetura Medallion (Bronze/Silver/Gold)

**Bronze Layer (Dados Brutos):**
- Armazenamento: MinIO bucket `bronze`
- Conteúdo: CSVs originais sem transformação
- Documentação: `docs/governance/bronze_layer.md`
- Versionamento: Controle de versão de todos os arquivos fonte

**Silver Layer (Dados Limpos):**
- Armazenamento: MinIO bucket `silver`
- Conteúdo: Dados limpos, validados e padronizados
- Transformações: Remoção de duplicatas, tratamento de valores nulos, normalização
- Documentação: `docs/governance/silver_layer.md`

**Gold Layer (Dados Prontos para Consumo):**
- Armazenamento: MinIO bucket `gold`
- Conteúdo: Dados otimizados para RAG e ML
- Formato: Embeddings vetoriais indexados no Milvus
- Documentação: `docs/governance/gold_layer.md`

### 2. Auditoria Completa de Inferências

**Metadados Rastreados em Cada Requisição:**

- `user_id`: Identificador do usuário que fez a requisição
- `inference_time_seconds`: Tempo total de inferência (segundos)
- `tokens_used`: Número total de tokens consumidos
- `chunks_retrieved`: Quantidade de documentos recuperados
- `collections_used`: Collections do Milvus consultadas
- `model_provider`: Provider do modelo (ollama, openai, anthropic, groq)
- `model_name`: Nome específico do modelo usado
- `top_p`: Parâmetro de amostragem nucleus utilizado
- `top_k`: Parâmetro top-k de amostragem utilizado
- `temperature`: Temperatura de geração utilizada

**Armazenamento:** Logs estruturados com rastreabilidade completa

### 3. Controle de Parâmetros de Geração

**Governança de Tokens:**
- Limite máximo por requisição: configurável via `max_tokens_per_request`
- Estimativa de custos em tempo real para modelos pagos
- Contabilização de tokens de entrada (prompt) e saída (resposta)

**Parâmetros Configuráveis:**
- `temperature` (0.0-2.0): Controla criatividade vs. determinismo
- `top_p` (0.0-1.0): Amostragem nucleus para diversidade controlada  
- `top_k` (1-100): Limita tokens candidatos para resposta focada

**Valores Padrão:**
- temperature: 0.2 (respostas técnicas e precisas)
- top_p: 0.9 (boa diversidade com controle)
- top_k: 40 (foco mantendo flexibilidade)

### 4. Versionamento de Embeddings

**Modelo de Embedding:**
- Nome: `paraphrase-multilingual-MiniLM-L12-v2`
- Dimensões: 384
- Idiomas: 50+ incluindo português brasileiro
- Provider: Sentence-Transformers

**Versionamento no Milvus:**
- Cada collection mantém histórico de embeddings
- Metadata de versão associada a cada documento
- Rastreabilidade completa de transformações

### 5. Documentação Automática no PostgreSQL

**Tabelas de Governança:**
- `pipeline_runs`: Histórico completo de execuções do pipeline
- `model_versions`: Versionamento de todos os modelos ML
- `data_quality_metrics`: Métricas de qualidade de cada camada
- `inference_logs`: Log detalhado de todas as inferências

### 6. Pool de Modelos Multi-Provider

**Modelos Disponíveis:**

**Ollama (Local - Gratuito):**
- llama3.2:1b (padrão)
- llama3.2:3b
- mistral:7b
- qwen2.5:3b

**OpenAI (Cloud - Pago):**
- gpt-4o
- gpt-4
- gpt-3.5-turbo

**Anthropic (Cloud - Pago):**
- claude-3-opus
- claude-3-sonnet

**Groq (Cloud - GRATUITO):**
- llama-3.1-70b-versatile
- mixtral-8x7b-32768
- gemma-7b-it

**Seleção Automática:** O sistema detecta automaticamente o provider correto baseado no nome do modelo fornecido.

## Datasets Utilizados

### 1. Histórico de Manutenção (📋)
- Origem: `data/vehicle_maintenance_data.csv`
- Registros: ~10.000
- Collection Milvus: `vehicle_maintenance`
- Conteúdo: Histórico real de manutenções, peças trocadas, custos

### 2. Sensores Preditivos (📊)
- Origem: `data/cars_hyundai.csv`
- Registros: ~1.000
- Collection Milvus: `car_predictive`
- Conteúdo: Leituras de sensores, temperatura, pressão, indicadores

### 3. Diagnóstico de Falhas (⚠️)
- Origem: `data/engine_fault_detection_dataset.csv`
- Registros: ~50.000
- Collection Milvus: `engine_fault`
- Conteúdo: Padrões de falhas, códigos OBD-II, diagnósticos técnicos

## Arquitetura do Sistema

### Componentes Principais:

1. **FastAPI** - API REST com documentação automática
2. **Milvus** - Banco vetorial para RAG multi-collection
3. **MinIO** - Object storage para arquitetura Medallion
4. **PostgreSQL** - Metadados e governança
5. **MLflow** - Rastreamento de experimentos ML
6. **Ollama** - LLM local para inferência
7. **Gradio** - Interface web interativa

### Fluxo de Inferência:

```
Usuário → FastAPI → RAGPipeline → Retriever (Milvus) → Generator (LLM) → Response + Metrics
```

### Tecnologias:

- **Python 3.11+**
- **Sentence-Transformers** para embeddings
- **Docker Compose** para orquestração
- **scikit-learn** para modelos ML
- **pandas** para processamento de dados
