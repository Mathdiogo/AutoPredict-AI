# 🎓 Demo para o Professor - AutoPredict AI

Este guia mostra como demonstrar todas as funcionalidades solicitadas pelo professor.

## ✅ Funcionalidades Implementadas

### 1. Pooling de Entry Point para Trocar Modelos
- ✅ Endpoint `GET /models` lista todos os modelos disponíveis
- ✅ Suporte a múltiplos providers: Ollama (local), OpenAI, Anthropic, Groq
- ✅ Troca dinâmica via parâmetro `model` no `POST /chat`
- ✅ Frontend com dropdown de seleção de modelo

### 2. Contrato da Aplicação / Governança
- ✅ Schema completo com `user_id`, `top_p`, `top_k`, `temperature`
- ✅ Validação via Pydantic nos endpoints
- ✅ Configurações de limites (max_tokens, defaults) no `config.py`
- ✅ Documentação de governança Medallion (Bronze/Silver/Gold)

### 3. Métricas no Response
- ✅ `inference_time_seconds` - Tempo de inferência em segundos
- ✅ `tokens_used` - Quantidade de tokens utilizados
- ✅ `chunks_retrieved` - Quantidade de chunks recuperados
- ✅ `collections_used` - Collections do Milvus consultadas
- ✅ `user_id` - ID do usuário da requisição
- ✅ `model_provider` e `model_name` - Modelo utilizado
- ✅ Parâmetros de geração (`top_p`, `top_k`, `temperature`)

### 4. Autoconhecimento (Perguntas Sobre o Sistema)
- ✅ "Quem é você?" → Identifica-se como AutoPredict AI
- ✅ "Quais modelos foram treinados?" → Lista Logistic Regression, Random Forest, XGBoost
- ✅ "Quais estratégias de governança?" → Explica Medallion, tokens, auditoria
- ✅ Entende variações das perguntas (semântica, não exato match)

---

## 🚀 Como Rodar para a Demo

### Passo 1: Subir os Serviços

```powershell
# Se ainda não estiver rodando
docker compose up -d

# Aguardar ~30s para todos os serviços iniciarem
docker compose ps
```

### Passo 2: Indexar System Info (Autoconhecimento)

```powershell
.\setup_system_info.ps1
```

Este script indexa o documento `SYSTEM_INFO.md` no Milvus para que o sistema possa responder perguntas sobre si mesmo via RAG.

### Passo 3: Testar Todos os Requisitos

```powershell
.\test_professor_requirements.ps1
```

Este script automatizado testa:
- ✅ Pooling de modelos
- ✅ Contrato/governança
- ✅ Métricas no response
- ✅ Pergunta "Quem é você?"
- ✅ Pergunta sobre modelos ML
- ✅ Pergunta sobre governança

**Se todos os testes passarem, o sistema está pronto para demonstração!**

---

## 🎯 Demonstração ao Vivo

### Opção 1: Frontend Gradio (Visual)

1. Acesse: http://localhost:7860

2. Selecione um modelo no dropdown (ex: `llama3.2:3b`)

3. Faça as perguntas do professor:

```
👤: Quem é você?
🤖: Sou AutoPredict AI, um sistema de manutenção preditiva...

👤: Quais modelos de machine learning foram treinados?
🤖: Treino 3 modelos: Logistic Regression, Random Forest e XGBoost...

👤: Como funciona a governança neste sistema?
🤖: Utilizamos arquitetura Medallion (Bronze/Silver/Gold)...
```

4. Ative "Mostrar fontes" para ver as **métricas de governança** completas:
   - ⏱️ Tempo de inferência
   - 🔢 Tokens utilizados
   - 📚 Chunks recuperados
   - 🗂️ Collections usadas
   - 👤 User ID
   - 🤖 Modelo usado

### Opção 2: API Direta (Swagger/Postman)

1. Acesse: http://localhost:8000/docs

2. **Demonstrar GET /models** (Pooling)
   ```json
   GET /models
   
   Resposta:
   {
     "models": [...],
     "default_model": "llama3.2:1b",
     "by_provider": {
       "ollama": 3,
       "openai": 3,
       "groq": 3,
       "anthropic": 3
     }
   }
   ```

3. **Demonstrar POST /chat** (Governança + Métricas)
   ```json
   POST /chat
   {
     "question": "Quem é você?",
     "user_id": "prof_demo",
     "model": "llama3.2:3b",
     "top_p": 0.9,
     "top_k": 40,
     "temperature": 0.2
   }
   
   Resposta:
   {
     "answer": "Sou AutoPredict AI...",
     "query": "Quem é você?",
     "sources": [...],
     "model": "ollama:llama3.2:3b",
     "total_docs_retrieved": 15,
     "metrics": {
       "inference_time_seconds": 3.45,
       "tokens_used": 1240,
       "chunks_retrieved": 15,
       "collections_used": ["vehicle_maintenance", "predictive_sensors", "engine_fault", "system_info"],
       "user_id": "prof_demo",
       "model_provider": "ollama",
       "model_name": "llama3.2:3b",
       "top_p": 0.9,
       "top_k": 40,
       "temperature": 0.2
     }
   }
   ```

---

## 📋 Checklist de Demonstração

### Antes da Apresentação
- [ ] Rodar `docker compose up -d`
- [ ] Aguardar todos os serviços ficarem "healthy" (~30s)
- [ ] Executar `.\setup_system_info.ps1` (indexar autoconhecimento)
- [ ] Executar `.\test_professor_requirements.ps1` (validar tudo)
- [ ] Abrir frontend em http://localhost:7860
- [ ] Abrir Swagger em http://localhost:8000/docs (em outra aba)

### Durante a Apresentação
1. **Mostrar Pooling de Modelos**
   - Abrir Swagger → GET /models
   - Mostrar lista de modelos Ollama + cloud

2. **Mostrar Contrato/Governança**
   - Swagger → POST /chat → "Try it out"
   - Mostrar campos: `user_id`, `top_p`, `top_k`, `temperature`
   - Executar uma pergunta

3. **Mostrar Métricas no Response**
   - Na resposta do POST /chat
   - Destacar o bloco `metrics` completo
   - Mostrar todos os campos solicitados

4. **Demonstrar Autoconhecimento**
   - No frontend Gradio (visual)
   - Perguntar: "Quem é você?"
   - Perguntar: "Quais modelos foram treinados e suas métricas?"
   - Perguntar: "Quais estratégias de governança este sistema usa?"
   - Ativar "Mostrar fontes" para exibir métricas na UI

5. **Destacar Semântica**
   - Fazer pergunta parecida mas não exata: "Me fala sobre os modelos de ML"
   - Sistema deve entender e responder corretamente

---

## 🐛 Troubleshooting

### API não está respondendo
```powershell
docker compose ps  # Verificar se todos estão "healthy"
docker compose logs api  # Ver logs do backend
```

### System Info não funciona
```powershell
# Reindexar
.\setup_system_info.ps1

# Verificar no Milvus
docker exec autopredict-api python -c "from pymilvus import utility; print(utility.list_collections())"
# Deve incluir 'system_info' na lista
```

### Modelo Ollama não responde
```powershell
# Pull do modelo se necessário
docker exec autopredict-ollama ollama pull llama3.2:3b

# Verificar se Ollama está ok
curl http://localhost:11434/api/tags
```

---

## 📚 Arquivos de Referência

- **Código da API**: `src/api/routes/chat.py`
- **Generator (multi-provider)**: `src/rag/generator.py`
- **Schemas (contrato)**: `src/api/schemas/chat.py`
- **Config (governança)**: `src/config.py`
- **System Info (autoconhecimento)**: `docs/governance/SYSTEM_INFO.md`
- **Governança Medallion**: `docs/governance/bronze_layer.md`, `silver_layer.md`, `gold_layer.md`

---

## 🎉 Conclusão

Todas as 4 funcionalidades solicitadas pelo professor foram implementadas:

1. ✅ **Pooling de modelos** - Troca dinâmica entre Ollama, OpenAI, Anthropic, Groq
2. ✅ **Contrato/governança** - `user_id`, `top_p`, `top_k`, limites de tokens, Medallion
3. ✅ **Métricas no response** - Tempo, tokens, chunks, collections, user_id
4. ✅ **Autoconhecimento** - Responde sobre si mesmo, modelos ML, governança (via RAG + prompt)

O sistema está **100% pronto para a demonstração ao professor**! 🚀
