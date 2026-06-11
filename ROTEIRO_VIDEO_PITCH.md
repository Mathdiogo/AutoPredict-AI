# 🎬 Roteiro para Vídeo Pitch - AutoPredict AI

**Duração estimada:** 8-12 minutos  
**Objetivo:** Demonstrar o sistema RAG respondendo às perguntas + mostrar arquitetura e governança

---

## 📋 Checklist Pré-Gravação

- [ ] Docker Desktop rodando
- [ ] Executar `docker compose up -d`
- [ ] Aguardar ~30s para todos os serviços subirem
- [ ] Executar `.\setup_system_info.ps1` (indexar autoconhecimento)
- [ ] Abrir terminais/abas:
  - Tab 1: Terminal no diretório do projeto
  - Tab 2: Frontend Gradio http://localhost:7860
  - Tab 3: Swagger UI http://localhost:8000/docs
  - Tab 4: MLflow http://localhost:5001 (opcional)
- [ ] Testar uma pergunta no frontend antes de gravar

---

## 🎥 ROTEIRO DO VÍDEO

### PARTE 1: INTRODUÇÃO (30 segundos)

**[Tela: Terminal ou VS Code aberto no projeto]**

> "Olá, sou [seu nome] e vou apresentar o AutoPredict AI, um sistema de manutenção preditiva automotiva que combina RAG - Retrieval-Augmented Generation - com Machine Learning para prever falhas antes que elas aconteçam."

> "O sistema integra múltiplos modelos LLM, 3 datasets especializados e governança completa de dados. Vamos ver tudo funcionando ao vivo."

---

### PARTE 2: VERIFICAR SERVIÇOS RODANDO (1 minuto)

**[Mostrar terminal]**

**Comando:**
```powershell
docker compose ps
```

**O que falar:**
> "Primeiro, vou mostrar que todos os serviços estão executando de forma isolada em containers Docker."

**[Enquanto mostra a saída]**

> "Temos aqui 7 serviços rodando:"
> - "**Postgres** - banco relacional para metadados e logs de ingestão"
> - "**Milvus** - banco vetorial para busca semântica"
> - "**MinIO** - object storage para arquitetura Medallion"
> - "**Ollama** - servidor LLM local rodando o Llama 3.2"
> - "**API** - backend FastAPI com todos os endpoints REST"
> - "**Frontend** - interface Gradio para demonstração"
> - "**MLflow** - rastreamento de experimentos de Machine Learning"

> "Todos com status 'healthy', rodando de forma isolada e orquestrados pelo Docker Compose."

**[Mostrar rapidamente o docker-compose.yml - scroll rápido]**

> "A arquitetura completa está definida neste docker-compose.yml com 7 serviços integrados."

---

### PARTE 3: RAG RESPONDENDO PERGUNTAS (4-5 minutos)

**[Abrir frontend Gradio http://localhost:7860]**

#### Pergunta 1: "Quem é você?" (1 min)

**[Selecionar modelo no dropdown]**

> "Vou usar o modelo local Llama 3.2 3B rodando no Ollama."

**[Digitar pergunta]**
```
Quem é você?
```

**[Ativar "Mostrar fontes" antes de enviar]**

**[Enquanto responde, comentar:]**

> "O sistema está buscando nos 3 datasets automotivos E na collection de autoconhecimento que indexamos."

**[Quando resposta aparecer, ler os pontos principais:]**

> "Veja que ele se identifica como AutoPredict AI, menciona que combina RAG com Machine Learning, e cita os múltiplos modelos LLM disponíveis."

**[Scroll até as métricas de governança]**

> "Aqui embaixo temos as **métricas de governança completas**:"
> - "Tempo de inferência: X segundos"
> - "Tokens utilizados: Y tokens"
> - "Chunks recuperados: Z documentos"
> - "Collections consultadas: vehicle_maintenance, predictive_sensors, engine_fault, **system_info**"
> - "User ID rastreado"
> - "Modelo usado: ollama:llama3.2:3b"
> - "Parâmetros: temperatura, top_p, top_k"

---

#### Pergunta 2: Modelos ML e Métricas (1.5 min)

**[Nova pergunta]**
```
Quais modelos de machine learning foram treinados e quais métricas foram utilizadas?
```

**[Enquanto responde:]**

> "Agora vou perguntar sobre os modelos de ML treinados."

**[Quando resposta aparecer, destacar:]**

> "Perfeito! O sistema lista os 3 algoritmos treinados:"
> - "**Logistic Regression** - modelo baseline"
> - "**Random Forest** - com acurácia de 95 a 98%"
> - "**XGBoost** - gradient boosting com acurácia de 96 a 99%"

> "E menciona as métricas utilizadas: acurácia, precisão, recall, F1-score, e que tudo é rastreado no MLflow."

**[Mostrar métricas de governança novamente]**

> "E novamente, todas as métricas de governança estão sendo rastreadas automaticamente."

---

#### Pergunta 3: Pré-processamento (1 min)

**[Nova pergunta]**
```
Qual foi o pré-processamento realizado nos dados?
```

**[Quando resposta aparecer:]**

> "O sistema explica o pipeline de dados em 3 camadas:"
> - "**Bronze** - dados brutos dos CSVs originais"
> - "**Silver** - dados limpos, sem duplicatas, valores nulos tratados"
> - "**Gold** - dados transformados em embeddings vetoriais e indexados no Milvus"

> "Essa é a arquitetura Medallion, uma das principais estratégias de governança que adotamos."

---

#### BÔNUS: Pergunta Técnica do Domínio (30 seg - opcional)

**[Fazer pergunta técnica automotiva para mostrar o RAG funcionando]**
```
Quais são as causas mais comuns de superaquecimento do motor?
```

**[Quando responder:]**

> "Veja que o sistema busca nos datasets automotivos reais e responde com contexto técnico, citando as fontes com os ícones dos datasets."

---

### PARTE 4: ENDPOINTS DA APLICAÇÃO (2 minutos)

**[Abrir Swagger UI http://localhost:8000/docs]**

**[Scroll pelos endpoints]**

> "Agora vou mostrar todos os endpoints REST da API."

#### 1. Health Check
**[Expandir GET /health]**

> "**GET /health** - verifica status de todos os serviços: Postgres, Milvus, MinIO, Ollama."

**[Click em "Try it out" → Execute]**

> "Veja que retorna 'healthy' e a contagem de documentos indexados em cada collection do Milvus."

---

#### 2. Models (Pooling)
**[Expandir GET /models]**

> "**GET /models** - este é o entry point do pooling de modelos. Lista todos os modelos disponíveis."

**[Execute]**

> "Retorna modelos Ollama locais, OpenAI, Anthropic e Groq, com o modelo padrão configurado."

---

#### 3. Chat (Principal)
**[Expandir POST /chat]**

> "**POST /chat** - este é o endpoint principal do RAG. Veja os parâmetros do contrato:"

**[Mostrar o schema na tela]**

> - "question - pergunta do usuário"
> - "**user_id** - para auditoria"
> - "**model** - para troca dinâmica de modelo"
> - "**top_p, top_k, temperature** - parâmetros de geração"
> - "min_score - score mínimo de relevância"

**[Try it out - fazer request]**

Request:
```json
{
  "question": "O que causa vibração no motor?",
  "user_id": "video_demo",
  "model": "llama3.2:3b",
  "top_p": 0.9,
  "top_k": 40,
  "temperature": 0.2
}
```

**[Mostrar response]**

> "E a resposta inclui:"
> - "answer - resposta gerada"
> - "sources - documentos usados como contexto"
> - "**metrics** - bloco completo de governança"

**[Scroll pelo bloco metrics]**

> "Aqui no metrics temos TODOS os campos solicitados: inference_time, tokens_used, chunks_retrieved, collections_used, user_id, model_provider, model_name, e os parâmetros."

---

#### 4. Streaming
**[Expandir GET /chat/stream]**

> "**GET /chat/stream** - versão com streaming, retorna tokens conforme são gerados. Também aceita o parâmetro model."

---

#### 5. Metadata
**[Expandir GET /metadata]**

> "**GET /metadata** - retorna informações sobre os datasets, embeddings e configurações do RAG."

---

#### 6. Examples
**[Expandir GET /chat/examples]**

> "**GET /chat/examples** - retorna perguntas de exemplo para facilitar testes."

---

### PARTE 5: ESTRATÉGIAS DE GOVERNANÇA (2 minutos)

**[Voltar para VS Code ou mostrar arquivos]**

> "Agora vou mostrar as estratégias de governança implementadas."

#### 1. Arquitetura Medallion

**[Mostrar estrutura de pastas ou MinIO UI se possível]**

> "Implementamos a **Arquitetura Medallion** com 3 camadas:"

**[Abrir docs/governance/bronze_layer.md rapidamente]**

> "**Bronze** - dados brutos no MinIO, bucket 'bronze'. CSVs originais sem transformação."

**[Abrir docs/governance/silver_layer.md]**

> "**Silver** - dados limpos no bucket 'silver'. Remoção de duplicatas, tratamento de nulos, validação de tipos."

**[Abrir docs/governance/gold_layer.md]**

> "**Gold** - dados prontos para consumo no bucket 'gold'. Embeddings vetoriais indexados no Milvus para busca semântica."

---

#### 2. Auditoria de Inferências

**[Mostrar src/api/schemas/chat.py - InferenceMetrics]**

> "Toda requisição é auditada com a classe **InferenceMetrics**:"
> - "user_id - quem fez a requisição"
> - "inference_time_seconds - quanto tempo levou"
> - "tokens_used - custo em tokens"
> - "chunks_retrieved - quantos documentos foram usados"
> - "collections_used - quais datasets foram consultados"
> - "model_provider e model_name - rastreabilidade do modelo"
> - "top_p, top_k, temperature - parâmetros usados"

---

#### 3. Versionamento MLflow

**[Abrir MLflow UI http://localhost:5001 se tiver tempo]**

> "Todos os modelos de ML são versionados no **MLflow**."

**[Mostrar experiments]**

> "Temos 3 experimentos, um para cada dataset:"
> - "AutoPredict-Maintenance"
> - "AutoPredict-Predictive"
> - "AutoPredict-EngineFault"

**[Click em um experiment - mostrar runs]**

> "Para cada dataset, treinamos 3 algoritmos: Logistic Regression, Random Forest e XGBoost."

**[Abrir um run - mostrar métricas]**

> "Cada run registra: parâmetros, métricas de acurácia, precisão, recall, F1-score, matriz de confusão, e o modelo treinado."

---

#### 4. Controle de Parâmetros

**[Mostrar src/config.py]**

> "Temos controle centralizado de **limites e parâmetros de governança** no config.py:"
> - "max_tokens_per_request: 1000"
> - "default_temperature, top_p, top_k"
> - "top_k_per_collection - quantos docs buscar por dataset"

---

#### 5. Documentação Completa

**[Mostrar pasta docs/governance/]**

> "Toda a governança está documentada na pasta docs/governance:"
> - "SYSTEM_INFO.md - informações completas do sistema"
> - "bronze_layer.md, silver_layer.md, gold_layer.md - documentação Medallion"
> - "GOVERNANCE_GUIDE.md - guia completo de governança"

---

### PARTE 6: ENCERRAMENTO (30 segundos)

**[Voltar para o frontend ou terminal]**

> "Recapitulando o que demonstrei:"
> 
> "✅ O RAG respondendo as 3 perguntas: quem é você, modelos ML, pré-processamento"
> 
> "✅ Todos os serviços rodando de forma isolada no Docker"
> 
> "✅ Os endpoints completos: health, models, chat, streaming, metadata"
> 
> "✅ As estratégias de governança: Medallion, auditoria de inferências, versionamento MLflow, controle de parâmetros"
>
> "O código completo está no GitHub, com documentação e scripts de setup. Obrigado!"

---

## 🎬 DICAS DE GRAVAÇÃO

### Preparação Técnica
- **Resolução:** 1920x1080 (Full HD)
- **FPS:** 30fps
- **Áudio:** Use fone de ouvido com microfone ou microfone USB
- **Gravador:** OBS Studio (gratuito) ou Loom

### Durante a Gravação
- **Fale devagar e claramente**
- **Pause 1-2 segundos entre tópicos** (facilita edição)
- **Se errar, pause e recomece a frase** (você pode cortar depois)
- **Não precisa ser perfeito** - naturalidade é mais importante
- **Mantenha cursor visível** quando clicar em coisas importantes

### Configuração de Tela
- **Feche abas desnecessárias**
- **Modo escuro no terminal e VS Code** (mais profissional)
- **Zoom in quando mostrar código** (Ctrl + para aumentar fonte)
- **Tela cheia em cada ferramenta** (Swagger, Frontend, etc)

### Ordem de Gravação (Opcional - grave em partes)
Você pode gravar cada parte separadamente e juntar depois:
1. Introdução
2. Docker ps
3. As 3 perguntas no RAG
4. Endpoints no Swagger
5. Governança (arquivos/docs)
6. Encerramento

Depois junta tudo no editor de vídeo (DaVinci Resolve é gratuito).

---

## 📦 CHECKLIST PÓS-GRAVAÇÃO

- [ ] Vídeo gravado (8-12 minutos)
- [ ] Upload no YouTube (pode ser não-listado)
- [ ] GitHub atualizado com todos os commits
- [ ] Preencher formulário com:
  - [ ] Link do vídeo
  - [ ] Link do GitHub

---

## 🆘 TROUBLESHOOTING

### Se algo não funcionar durante a gravação:

**API não responde:**
```powershell
docker compose restart api
# Aguardar 10s
curl http://localhost:8000/health
```

**System info não responde perguntas:**
```powershell
.\setup_system_info.ps1
# Aguardar completar
```

**Frontend com erro:**
```powershell
docker compose logs frontend
```

**Ollama lento:**
> "O Ollama pode demorar na primeira resposta porque está carregando o modelo. Isso é normal."

---

## 🎯 RESUMO - O QUE MOSTRAR

| Requisito | Onde Mostrar | Tempo |
|-----------|--------------|-------|
| Quem é você? | Frontend Gradio | 1 min |
| Modelos ML e métricas | Frontend Gradio | 1.5 min |
| Pré-processamento | Frontend Gradio | 1 min |
| Serviços isolados | Terminal `docker compose ps` | 1 min |
| Endpoints | Swagger UI | 2 min |
| Governança | Docs + código + MLflow | 2 min |

**Total:** ~8-9 minutos + intro/encerramento = **10-12 minutos**

---

Boa sorte com a gravação! 🎬🚀
