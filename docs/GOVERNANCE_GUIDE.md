# Guia de Governança e Métricas - AutoPredict AI

## 📊 Novas Funcionalidades de Governança

Este guia explica as novas funcionalidades de governança e métricas implementadas no AutoPredict AI.

## 🎯 1. Pooling de Modelos Multi-Provider

### Modelos Disponíveis

O sistema agora suporta múltiplos providers de LLM:

**Ollama (Local - Gratuito):**
- `llama3.2:1b` (padrão)
- `llama3.2:3b`
- `mistral:7b`
- `qwen2.5:3b`

**OpenAI (Cloud - Pago):**
- `gpt-4o`
- `gpt-4`
- `gpt-3.5-turbo`

**Anthropic (Cloud - Pago):**
- `claude-3-opus`
- `claude-3-sonnet`

**Groq (Cloud - GRATUITO):**
- `llama-3.1-70b-versatile`
- `mixtral-8x7b-32768`
- `gemma-7b-it`

### Como Usar

```bash
# Requisição com modelo específico
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quais são as causas de superaquecimento?",
    "model": "llama3.2:3b",
    "user_id": "user123",
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40
  }'
```

### Listar Modelos Disponíveis

```bash
curl http://localhost:8000/models
```

## 📈 2. Métricas de Inferência

### Response com Métricas Completas

Toda resposta agora inclui um objeto `metrics` com:

```json
{
  "answer": "Resposta do sistema...",
  "query": "Pergunta original",
  "sources": [...],
  "model": "ollama:llama3.2:3b",
  "total_docs_retrieved": 15,
  "metrics": {
    "inference_time_seconds": 2.45,
    "tokens_used": 856,
    "chunks_retrieved": 15,
    "collections_used": [
      "vehicle_maintenance",
      "engine_fault",
      "system_info"
    ],
    "user_id": "user123",
    "model_provider": "ollama",
    "model_name": "llama3.2:3b",
    "top_p": 0.9,
    "top_k": 40,
    "temperature": 0.2
  }
}
```

### Campos de Métricas

| Campo | Descrição |
|-------|-----------|
| `inference_time_seconds` | Tempo total de inferência em segundos |
| `tokens_used` | Número total de tokens utilizados |
| `chunks_retrieved` | Quantidade de documentos recuperados |
| `collections_used` | Collections do Milvus consultadas |
| `user_id` | ID do usuário que fez a requisição |
| `model_provider` | Provider do modelo (ollama, openai, etc) |
| `model_name` | Nome do modelo utilizado |
| `top_p` | Parâmetro de amostragem nucleus |
| `top_k` | Parâmetro de amostragem top-k |
| `temperature` | Temperatura de geração |

## 🔐 3. Controle de Parâmetros

### Parâmetros Configuráveis

#### Temperature (0.0 - 2.0)
- **0.0-0.3**: Respostas determinísticas e precisas (ideal para diagnóstico técnico)
- **0.4-0.7**: Balanceado entre precisão e criatividade
- **0.8-2.0**: Respostas mais criativas e variadas

#### Top-P (0.0 - 1.0)
- **0.9**: Padrão - boa diversidade com controle
- **0.95**: Mais diversidade
- **0.8**: Mais focado

#### Top-K (1 - 100)
- **40**: Padrão - bom balanceamento
- **20**: Mais focado e determinístico
- **60-80**: Mais variação

### Configuração no `.env`

```env
# Governança e Tokens
MAX_TOKENS_PER_REQUEST=1000
DEFAULT_TEMPERATURE=0.2
DEFAULT_TOP_P=0.9
DEFAULT_TOP_K=40
```

## 🤖 4. Sistema de Autoconhecimento

### Perguntas sobre o Próprio Sistema

O sistema agora pode responder perguntas sobre si mesmo:

**Exemplos de perguntas suportadas:**

1. **Identidade:**
   - "Quem é você?"
   - "O que você faz?"
   - "Qual é a sua função?"

2. **Modelos de ML:**
   - "Quais modelos foram treinados?"
   - "Quais métricas dos modelos?"
   - "Como funciona o Random Forest?"

3. **Governança:**
   - "Quais estratégias de governança você usa?"
   - "Como funciona a arquitetura Medallion?"
   - "O que é registrado nas métricas?"

4. **Datasets:**
   - "Quais dados você utiliza?"
   - "Quantas collections você tem?"
   - "De onde vêm os dados?"

### Indexar Informações do Sistema

Para que o sistema responda essas perguntas, execute:

```powershell
# Windows
.\index_system_info.ps1

# Ou diretamente
python src\data_pipeline\index_system_info.py
```

## 📝 5. Exemplo Completo de Uso

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "question": "Quem é você e quais modelos você tem?",
        "user_id": "user123",
        "model": "llama3.2:3b",
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "min_score": 0.25
    }
)

data = response.json()

print(f"Resposta: {data['answer']}")
print(f"\nMétricas:")
print(f"  Tempo: {data['metrics']['inference_time_seconds']:.2f}s")
print(f"  Tokens: {data['metrics']['tokens_used']}")
print(f"  Chunks: {data['metrics']['chunks_retrieved']}")
print(f"  Collections: {', '.join(data['metrics']['collections_used'])}")
```

### JavaScript/Frontend

```javascript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: "Quem é você?",
    user_id: "user123",
    model: "llama3.2:3b",
    temperature: 0.2,
    top_p: 0.9,
    top_k: 40
  })
});

const data = await response.json();

console.log('Resposta:', data.answer);
console.log('Tempo de inferência:', data.metrics.inference_time_seconds);
console.log('Tokens usados:', data.metrics.tokens_used);
```

## 🔍 6. Auditoria e Logs

### Logs Estruturados

Todas as requisições são logadas com:

```
[API] POST /chat - 'Quem é você?' (user: user123)
[RAG] Nova query: 'Quem é você?'
[Retriever] Query: 'Quem é você?' (buscando k=5 por coleção)
[Retriever] system_info: 10 candidatos
[Retriever] MMR: 40 candidatos → 20 selecionados
[Generator] Provider: ollama, Model: llama3.2:3b
[Generator] Resposta gerada (1243 chars, 311 tokens, 2.34s)
[RAG] Resposta gerada (20 docs, 311 tokens, 2.34s)
```

### Rastreamento de Custos

Para modelos pagos (OpenAI, Anthropic), os tokens são contabilizados:

```python
# Estimativa de custo OpenAI
tokens_used = data['metrics']['tokens_used']
cost_gpt4 = (tokens_used / 1000) * 0.03  # $0.03 per 1K tokens
cost_gpt35 = (tokens_used / 1000) * 0.002  # $0.002 per 1K tokens
```

## 🚀 7. Testes das Novas Funcionalidades

### Teste 1: Pooling de Modelos

```bash
# Testa diferentes modelos
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Teste", "model": "llama3.2:1b"}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Teste", "model": "llama3.2:3b"}'
```

### Teste 2: Métricas

```bash
# Verifica métricas na resposta
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Teste", "user_id": "test_user"}' \
  | jq '.metrics'
```

### Teste 3: Autoconhecimento

```bash
# Testa perguntas sobre o sistema
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Quem é você?"}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Quais modelos foram treinados?"}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Quais estratégias de governança você usa?"}'
```

## 📚 8. Documentação Técnica

### Arquivos Relacionados

- `src/api/schemas/chat.py` - Schemas com métricas
- `src/rag/generator.py` - Geração com medição de tempo/tokens
- `src/rag/pipeline.py` - Pipeline com métricas
- `src/rag/retriever.py` - Busca incluindo system_info
- `src/config.py` - Configurações de governança
- `docs/governance/SYSTEM_INFO.md` - Knowledge base do sistema
- `src/data_pipeline/index_system_info.py` - Indexação no Milvus

### Próximos Passos

1. ✅ Pooling de modelos implementado
2. ✅ Métricas completas de inferência
3. ✅ Sistema de autoconhecimento
4. ✅ Controle de parâmetros (top_p, top_k, temperature)
5. ✅ Auditoria completa com user_id e collections
6. 🔄 Dashboard de métricas (futuro)
7. 🔄 Rate limiting por user_id (futuro)
8. 🔄 Cache de embeddings (futuro)

## 🎓 9. Para Apresentação ao Professor

### Demonstrações Sugeridas

1. **Pooling de Modelos:**
   ```bash
   # Mostra lista de modelos disponíveis
   curl http://localhost:8000/models
   
   # Troca entre modelos em tempo real
   curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"question": "Teste", "model": "llama3.2:1b"}'
   ```

2. **Métricas e Governança:**
   ```bash
   # Mostra métricas completas
   curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"question": "Teste", "user_id": "demo_prof"}' | jq '.metrics'
   ```

3. **Autoconhecimento:**
   ```bash
   # Sistema responde sobre si mesmo
   curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"question": "Quem é você e quais modelos você treinou?"}'
   ```

### Pontos-Chave a Destacar

✅ **Pooling de modelos** - Troca dinâmica entre LLMs  
✅ **Governança** - Auditoria completa (user_id, tokens, tempo, collections)  
✅ **Contrato da aplicação** - Parâmetros configuráveis (top_k, top_p, temperature)  
✅ **Métricas de inferência** - Tempo, tokens, chunks recuperados  
✅ **Autoconhecimento** - Sistema responde sobre modelos, governança e arquitetura  
✅ **Multi-collection RAG** - 4 datasets incluindo informações do próprio sistema  
