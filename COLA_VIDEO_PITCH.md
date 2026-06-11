# 📝 COLA - Guia Rápido para Gravação

**Imprima ou tenha em outra tela enquanto grava!**

---

## ⚡ ORDEM DO VÍDEO

1. **INTRO** (30s) - "AutoPredict AI, RAG + ML, manutenção preditiva"

2. **DOCKER PS** (1min)
   ```powershell
   docker compose ps
   ```
   Citar: 7 serviços (Postgres, Milvus, MinIO, Ollama, API, Frontend, MLflow)

3. **PERGUNTAS RAG** (4min)
   - Frontend: http://localhost:7860
   - ✅ "Quem é você?"
   - ✅ "Quais modelos de ML foram treinados e quais métricas?"
   - ✅ "Qual foi o pré-processamento dos dados?"
   - **SEMPRE ativar "Mostrar fontes"**
   - **SEMPRE comentar as métricas de governança**

4. **ENDPOINTS** (2min)
   - Swagger: http://localhost:8000/docs
   - GET /health
   - GET /models
   - POST /chat (mostrar schema + fazer request)
   - GET /chat/stream
   - GET /metadata

5. **GOVERNANÇA** (2min)
   - Medallion (Bronze/Silver/Gold) - mostrar docs/governance/
   - InferenceMetrics - mostrar src/api/schemas/chat.py
   - MLflow - http://localhost:5001 (experiments + runs)
   - Config - mostrar src/config.py

6. **ENCERRAMENTO** (30s) - Recapitular os 4 pontos

---

## 🎤 FRASES-CHAVE

### Intro
> "AutoPredict AI combina RAG com Machine Learning para manutenção preditiva automotiva."

### Docker
> "Todos os 7 serviços estão executando de forma isolada em containers Docker."

### RAG - Após cada resposta
> "Veja as métricas de governança: tempo, tokens, chunks, collections, user_id, modelo."

### Endpoints
> "O POST /chat é o endpoint principal, com contrato completo: user_id, model, top_p, top_k, temperature."

### Governança
> "Implementamos 4 estratégias: Medallion, auditoria de inferências, versionamento MLflow, controle de parâmetros."

---

## 📋 CHECKLIST ANTES DE GRAVAR

- [ ] `docker compose up -d`
- [ ] `.\setup_system_info.ps1`
- [ ] Abrir 4 abas:
  - [ ] Terminal
  - [ ] Frontend (7860)
  - [ ] Swagger (8000/docs)
  - [ ] MLflow (5001)
- [ ] Fechar abas desnecessárias
- [ ] Testar uma pergunta no frontend
- [ ] Modo escuro em tudo
- [ ] Microfone testado

---

## 🎯 MÉTRICAS DE GOVERNANÇA (sempre mencionar)

- ⏱️ inference_time_seconds
- 🔢 tokens_used
- 📚 chunks_retrieved
- 🗂️ collections_used
- 👤 user_id
- 🤖 model_provider:model_name
- 🎛️ top_p, top_k, temperature

---

## 🔧 COMANDOS RÁPIDOS

**Se API travar:**
```powershell
docker compose restart api
```

**Se Ollama travar:**
```powershell
docker compose restart ollama
```

**Ver logs:**
```powershell
docker compose logs api --tail=50
```

---

## ⏱️ TIMING

- Intro: 30s
- Docker ps: 1min
- 3 Perguntas RAG: 4min (1min, 1.5min, 1min + 30s bônus)
- Endpoints: 2min
- Governança: 2min
- Encerramento: 30s

**TOTAL: ~10 minutos**

---

## 💡 DICAS DE OURO

1. **Fale devagar** - melhor sobrar tempo que correr
2. **Pause entre tópicos** - facilita edição
3. **Se errar, recomece a frase** - não precisa gravar tudo de novo
4. **Zoom no código** - Ctrl + aumenta fonte
5. **Cursor visível** - mostre onde está clicando
6. **Naturalidade > Perfeição** - você sabe do que está falando!

---

## 📤 APÓS GRAVAR

1. Upload YouTube (pode ser não-listado)
2. Atualizar GitHub
3. Preencher formulário: https://forms.office.com/r/asFgrfG6yT

---

**BOA SORTE! 🎬🚀**
