# 🚀 Guia Rápido - Novas Funcionalidades de Governança

## Melhorias Implementadas

✅ **Pooling de modelos** - Troca entre diferentes LLMs (Ollama, OpenAI, Anthropic, Groq)  
✅ **Métricas completas** - Tempo, tokens, chunks, collections, user_id  
✅ **Governança de tokens** - Parâmetros top_k, top_p, temperature configuráveis  
✅ **Autoconhecimento** - Sistema responde sobre si mesmo, modelos treinados e governança  

---

## 📋 Setup Rápido

### 1. Certifique-se que os serviços estão rodando

```powershell
docker compose up -d
```

### 2. Indexe as informações do sistema

```powershell
.\index_system_info.ps1
```

### 3. Teste as funcionalidades

```powershell
.\test_governance.ps1
```

---

## 🧪 Testes Manuais

### Teste 1: Listar Modelos Disponíveis

```bash
curl http://localhost:8000/models
```

### Teste 2: Requisição com Métricas Completas

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quais são as causas de superaquecimento?",
    "user_id": "user123",
    "model": "llama3.2:3b",
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40
  }'
```

### Teste 3: Perguntas de Autoconhecimento

```bash
# Quem é você?
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Quem é você?"}'

# Modelos treinados
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Quais modelos foram treinados e suas métricas?"}'

# Governança
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Quais estratégias de governança você usa?"}'
```

---

## 📊 Response com Métricas

Agora toda resposta inclui:

```json
{
  "answer": "Resposta do LLM...",
  "query": "Pergunta original",
  "sources": [...],
  "model": "ollama:llama3.2:3b",
  "total_docs_retrieved": 15,
  "metrics": {
    "inference_time_seconds": 2.45,
    "tokens_used": 856,
    "chunks_retrieved": 15,
    "collections_used": ["vehicle_maintenance", "engine_fault", "system_info"],
    "user_id": "user123",
    "model_provider": "ollama",
    "model_name": "llama3.2:3b",
    "top_p": 0.9,
    "top_k": 40,
    "temperature": 0.2
  }
}
```

---

## 🎬 Roteiro para Demonstração

### 1. Pooling de Modelos (2 min)

```powershell
# Mostrar modelos disponíveis
curl http://localhost:8000/models

# Trocar modelo
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{"question": "Teste", "model": "llama3.2:1b"}'
```

### 2. Métricas e Governança (3 min)

```powershell
# Requisição completa com métricas
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{
    "question": "Causas de superaquecimento?",
    "user_id": "demo_prof",
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40
  }'
```

Destacar na resposta:
- `inference_time_seconds`
- `tokens_used`
- `chunks_retrieved`
- `collections_used`
- `user_id`

### 3. Autoconhecimento (3 min)

```powershell
# Pergunta 1: Identidade
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{"question": "Quem é você?"}'

# Pergunta 2: Modelos
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{"question": "Quais modelos foram treinados?"}'

# Pergunta 3: Governança
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{"question": "Mostre as estratégias de governança"}'
```

---

## 📁 Arquivos Criados/Modificados

### Novos Arquivos:
- `docs/governance/SYSTEM_INFO.md` - Knowledge base do sistema
- `docs/GOVERNANCE_GUIDE.md` - Guia completo de uso
- `docs/CHANGES_SUMMARY.md` - Resumo técnico das mudanças
- `src/data_pipeline/index_system_info.py` - Script de indexação
- `index_system_info.ps1` - Helper de indexação
- `test_governance.ps1` - Script de testes
- `docs/QUICK_START_GOVERNANCE.md` - Este arquivo

### Arquivos Modificados:
- `src/api/schemas/chat.py` - Schemas com métricas
- `src/config.py` - Parâmetros de governança
- `src/rag/generator.py` - Medição de tempo/tokens
- `src/rag/pipeline.py` - Agregação de métricas
- `src/rag/retriever.py` - Busca em system_info
- `src/api/routes/chat.py` - Response com métricas

---

## ✅ Checklist Antes da Apresentação

- [ ] Docker services running (`docker compose ps`)
- [ ] Sistema indexado (`.\index_system_info.ps1`)
- [ ] API respondendo (`curl http://localhost:8000/health`)
- [ ] Testes passando (`.\test_governance.ps1`)

---

## 📚 Documentação Completa

Para mais detalhes, consulte:

- **Guia de Uso:** `docs/GOVERNANCE_GUIDE.md`
- **Resumo Técnico:** `docs/CHANGES_SUMMARY.md`
- **Knowledge Base:** `docs/governance/SYSTEM_INFO.md`

---

## 🎯 Pontos-Chave

1. ✅ Pooling de modelos implementado
2. ✅ Governança completa (user_id, tokens, tempo)
3. ✅ Contrato robusto (top_k, top_p, temperature)
4. ✅ Métricas detalhadas
5. ✅ Autoconhecimento funcional
6. ✅ RAG com 4 collections

**Sistema pronto para apresentação! 🚀**
