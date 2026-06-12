# Guia Rápido — Vídeo Pitch (Entrega Final)

**Tempo alvo:** 5–7 minutos  
**Foco:** só o que o professor pediu na mensagem. Arquitetura geral e sprints já foram apresentados.

---

## Antes de gravar (5 min)

```powershell
cd AutoPredict-AI
docker compose up -d
# Aguardar ~30s
docker ps --filter "name=autopredict"
```

Abrir no navegador:
- **Chat:** http://localhost:7860
- **Swagger:** http://localhost:8000/docs
- **MinIO:** http://localhost:9001 (`minioadmin` / `minioadmin123`)

No chat, selecionar **Llama 3.3 70B (Groq)** — respostas rápidas no vídeo.

Testar as 3 perguntas uma vez antes de gravar (Ctrl+F5 no frontend).

---

## Roteiro do vídeo

### 1. Serviços isolados (~1 min)

**Mostrar terminal:**

```powershell
docker ps --filter "name=autopredict"
```

**Falar (curto):**
> "Cada serviço roda isolado em container Docker, orquestrado pelo Compose do projeto AutoPredict."

| Container | Função (1 frase) |
|---|---|
| autopredict-api | API REST (RAG + ML) |
| autopredict-frontend | Interface Gradio |
| autopredict-ollama | LLM local (GPU) |
| autopredict-milvus | Banco vetorial (RAG) |
| autopredict-minio | Data Lake (Medallion + governança) |
| autopredict-postgres | Metadados e logs |
| autopredict-mlflow | Experimentos ML |
| autopredict-etcd | Dependência do Milvus |

> "Projeto com `name: autopredict` no compose — não conflita com outros containers da máquina."

*(Opcional: mostrar só os 8 autopredict, ignorar containers de outros projetos.)*

---

### 2. RAG — 3 perguntas obrigatórias (~2–3 min)

**Tela:** http://localhost:7860  
**Modelo:** Groq Llama 3.3 70B (ou Ollama local se preferir offline)

Fazer **uma pergunta por vez**. Não misturar.

#### Pergunta 1
```
Quem é você?
```
**O que mostrar:** resposta sobre AutoPredict AI (RAG + ML + governança). **Sem** análise de veículo.

#### Pergunta 2
```
Quais modelos foram treinados e quais métricas foram usadas?
```
**O que mostrar:** Logistic Regression, Random Forest, XGBoost × 3 datasets; métricas Accuracy, F1, AUC; MLflow http://localhost:5001 (aba rápida se quiser).

#### Pergunta 3
```
Qual o pré-processamento realizado nos dados?
```
**O que mostrar:** pipeline Medallion — Bronze (bruto) → Silver (limpeza, tipos, duplicatas) → Gold (embeddings no Milvus).

**Dica:** marque **"Mostrar documentos"** só em pergunta veicular (ex.: pressão dos pneus). Nas 3 meta-perguntas, deixe desmarcado — resposta direta, sem fontes de carro.

---

### 3. Endpoints da API (~1,5 min)

**Tela:** http://localhost:8000/docs

**Falar antes de scrollar:**
> "A API REST foi feita em FastAPI. Cada rota tem contrato Pydantic — request e response validados automaticamente. Documentação interativa aqui no Swagger."

---

#### Chat (núcleo do RAG)

| Método | Endpoint | Explicação |
|---|---|---|
| `POST` | `/chat` | **Endpoint principal.** Recebe a pergunta em JSON (`question`, `model`, `min_score`, `temperature`, `user_id`…). Internamente: busca documentos similares no Milvus (3 datasets) → monta contexto → envia ao LLM → devolve resposta completa com **fontes usadas** e bloco **metrics** (tokens, tempo de inferência, collections consultadas). Ideal para integrações e testes via Postman. |
| `GET` | `/chat/stream` | Mesmo fluxo RAG, mas a resposta chega **token a token** (Server-Sent Events). É o que o frontend Gradio usa para parecer que o bot está “digitando” em tempo real. Parâmetros via query string: `?question=...&model=...` |
| `GET` | `/chat/examples` | Retorna lista de perguntas de exemplo. O frontend consome isso para montar os chips de sugestão na interface — separa conteúdo da UI da lógica da API. |

---

#### Observabilidade e operação

| Método | Endpoint | Explicação |
|---|---|---|
| `GET` | `/health` | **Health check** dos serviços críticos: Milvus, PostgreSQL e Ollama (true/false). Também retorna quantos documentos estão indexados por collection (`vehicle_maintenance`, `car_predictive`, `engine_fault`). Serve para monitorar se o RAG está pronto antes de demo/produção. |

---

#### Modelos e configuração

| Método | Endpoint | Explicação |
|---|---|---|
| `GET` | `/models` | Lista todos os LLMs disponíveis, agrupados por provider: **Ollama** (local), **Groq** (cloud grátis), OpenAI e Anthropic. Indica qual exige API key e qual é o modelo padrão. Alimenta o dropdown “Modelo LLM” no Gradio. |

---

#### Metadados e governança

| Método | Endpoint | Explicação |
|---|---|---|
| `GET` | `/metadata` | Visão **consolidada** do sistema: os 3 datasets (nome, descrição, qtd. de docs), config de embeddings (modelo, dimensões), config RAG (`top_k`, LLM padrão) e status geral (`operational` / `degraded`). |
| `GET` | `/metadata/datasets` | Detalhe **por dataset** — útil para auditoria: quantos registros foram indexados em cada collection do Milvus e o que cada base representa (manutenção, sensores, falhas de motor). |
| `GET` | `/metadata/config` | Expõe configurações técnicas ativas: URL do LLM, modelo de embedding, dimensão dos vetores, parâmetros padrão de geração. Transparência para quem integra ou audita o sistema. |

---

**Falar ao fechar o Swagger:**
> "Resumindo: `/chat` e `/chat/stream` executam o RAG; `/health` e `/metadata` garantem observabilidade; `/models` permite trocar o LLM sem mudar código. Toda resposta do chat traz métricas de governança — tokens, tempo, chunks e provider usados."

*(Demo rápida: expandir `GET /health` → **Try it out** → **Execute** e mostrar os 11.100 docs indexados.)*

---

### 4. Governança (~1 min)

**Mostrar MinIO:** http://localhost:9001 → bucket **`governance`** → pasta **`governance/`**

Arquivos:
- `bronze_layer.md`, `silver_layer.md`, `gold_layer.md`
- `README.md`, `SYSTEM_INFO.md`

**Falar:**
> "Adotamos arquitetura Medallion: Bronze, Silver e Gold no MinIO para os dados; bucket separado **governance** para a documentação das estratégias. MLflow registra os 9 modelos ML; PostgreSQL guarda logs de ingestão. Cada inferência na API registra provider, tokens e collections consultadas."

**Estratégias (bullet mental):**
1. Medallion (Bronze / Silver / Gold)
2. Documentação versionada no MinIO + MLflow
3. Auditoria por requisição (métricas no `/chat`)
4. Isolamento Docker (`autopredict_`)

---

### 5. Fechamento (~20 s)

> "Sistema RAG respondendo às três perguntas, serviços isolados, endpoints documentados e governança no MinIO e MLflow. Repositório e este vídeo serão enviados no formulário pelo PO."

---

## Checklist pós-gravação

- [ ] Vídeo mostra `docker ps` (autopredict)
- [ ] 3 perguntas RAG respondidas no chat
- [ ] Swagger com endpoints visível
- [ ] MinIO bucket `governance` mostrado
- [ ] PO preencheu o formulário (link vídeo + GitHub atualizado)

---

## Comandos de emergência

```powershell
docker compose restart api frontend   # chat travou
docker compose ps                      # ver status
Invoke-RestMethod http://localhost:8000/health   # API ok?
```
