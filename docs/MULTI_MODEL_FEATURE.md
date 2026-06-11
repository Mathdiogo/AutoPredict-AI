# 🤖 Multi-Modelo LLM - Nova Funcionalidade

## ✅ O que foi implementado?

Adicionamos suporte para **alternar entre múltiplos modelos LLM**, incluindo:

### 1. **Modelos Locais (Ollama)** 🏠
- llama3.2:3b (padrão)
- mistral
- qwen2.5:3b
- Qualquer modelo instalado no Ollama

### 2. **Modelos Cloud (BONUS)** ☁️
- **OpenAI**: GPT-4, GPT-4o, GPT-3.5-turbo
- **Anthropic**: Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku

---

## 🔧 Arquivos Modificados

### **Backend (API)**

#### 1. `src/config.py`
- Adicionado: `openai_api_key` e `anthropic_api_key` (opcional)

#### 2. `src/rag/generator.py` (REESCRITO)
- **Multi-provider support**: detecta automaticamente qual provider usar
- **3 métodos de geração**:
  - `_generate_ollama()` - local via Ollama
  - `_generate_openai()` - OpenAI API
  - `_generate_anthropic()` - Anthropic API
- **Streaming**: funciona apenas com Ollama (cloud retorna resposta completa)

#### 3. `src/rag/pipeline.py`
- Métodos `query()` e `stream_query()` agora aceitam parâmetro `model`

#### 4. `src/api/schemas/chat.py`
- Campo `model: Optional[str]` adicionado em `ChatRequest`

#### 5. `src/api/routes/chat.py`
- Passa `model` do request para o pipeline

#### 6. `src/api/routes/models.py` (NOVO)
- **GET /models**: retorna lista de modelos disponíveis
- Consulta Ollama para modelos locais
- Lista modelos cloud se API keys estiverem configuradas

#### 7. `src/api/main.py`
- Registrado router de `models`

### **Frontend (Gradio)**

#### 8. `src/frontend/app.py`
- **Função `get_available_models()`**: busca modelos da API
- **Dropdown de seleção**: permite escolher modelo na interface
- **Função `chat_with_api()`**: aceita `selected_model` e envia para API
- **Detecção automática**: modelos cloud usam POST (sem streaming), Ollama usa GET /stream

---

## 🚀 Como usar?

### 1. **Modelos Locais (Ollama)** - Já funciona!

Basta selecionar no dropdown do frontend. Para adicionar mais modelos:

```bash
# Exemplo: instalar Mistral
docker exec autopredict-ollama ollama pull mistral

# Exemplo: instalar Qwen 2.5
docker exec autopredict-ollama ollama pull qwen2.5:3b
```

### 2. **Modelos Cloud (OpenAI/Anthropic)** - BONUS

#### Passo 1: Adicionar API Keys no `.env`

```bash
# Opcional - OpenAI
OPENAI_API_KEY=sk-proj-...

# Opcional - Anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

#### Passo 2: Reiniciar API

```bash
docker compose restart api frontend
```

#### Passo 3: Selecionar modelo no dropdown

Os modelos aparecerão automaticamente se as keys estiverem configuradas!

---

## 🧪 Testando

### 1. Listar modelos disponíveis

```bash
curl http://localhost:8000/models | jq
```

**Resposta esperada:**
```json
{
  "models": [
    {
      "name": "llama3.2:3b",
      "display_name": "llama3.2:3b",
      "provider": "ollama",
      "local": true
    },
    {
      "name": "gpt-4",
      "display_name": "GPT-4 (OpenAI)",
      "provider": "openai",
      "local": false
    }
  ],
  "default_model": "llama3.2:3b",
  "total_available": 2
}
```

### 2. Chat com modelo específico

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quais falhas são comuns em veículos com mais de 100.000km?",
    "model": "mistral"
  }'
```

### 3. Frontend (Gradio)

1. Acesse: http://localhost:7860
2. Veja o novo dropdown **"🤖 Modelo LLM"**
3. Escolha entre:
   - **Local (Ollama)**: streaming em tempo real ⚡
   - **Cloud (OpenAI/Anthropic)**: resposta completa de uma vez

---

## 📊 Comparação de Modelos

| Modelo | Provider | Local | Streaming | Velocidade | Custo |
|--------|----------|-------|-----------|------------|-------|
| llama3.2:3b | Ollama | ✅ | ✅ | ~50 tok/s | Grátis |
| mistral | Ollama | ✅ | ✅ | ~60 tok/s | Grátis |
| gpt-4o | OpenAI | ❌ | ❌ | Rápido | $0.005/1K tokens |
| claude-3-sonnet | Anthropic | ❌ | ❌ | Rápido | $0.003/1K tokens |

---

## 🎯 Valor Acadêmico

Esta funcionalidade demonstra:

1. **Arquitetura Multi-Provider**: design pattern para abstrair diferentes APIs
2. **Feature Flags**: modelos cloud só aparecem se configurados
3. **Fallback Gracioso**: se API não responde, usa modelo local
4. **UX Responsiva**: detecção automática de streaming vs. resposta completa
5. **Extensibilidade**: fácil adicionar novos providers (ex: Cohere, Gemini)

---

## 📝 Endpoints da API

### GET /models
Lista todos os modelos disponíveis.

**Response:**
```json
{
  "models": [...],
  "default_model": "llama3.2:3b",
  "total_available": 1,
  "by_provider": {
    "ollama": 1,
    "openai": 0,
    "anthropic": 0
  }
}
```

### POST /chat
Agora aceita campo `model` opcional.

**Request:**
```json
{
  "question": "Meu motor superaquece, o que fazer?",
  "model": "gpt-4",
  "min_score": 0.25
}
```

**Response:**
```json
{
  "answer": "Com base nos dados de sensores...",
  "query": "Meu motor superaquece, o que fazer?",
  "sources": [...],
  "model": "openai:gpt-4",
  "total_docs_retrieved": 15
}
```

---

## ✅ Status Final

- ✅ Backend multi-provider funcionando
- ✅ Endpoint /models implementado
- ✅ Frontend com dropdown de seleção
- ✅ Streaming funcional (Ollama)
- ✅ Modelos cloud funcionais (se API keys configuradas)
- ✅ Detecção automática de provider
- ✅ Tratamento de erros (API key inválida, rate limits, etc.)

**PRONTO PARA DEMONSTRAÇÃO ACADÊMICA! 🎓**
