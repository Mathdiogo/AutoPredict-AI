# Guia Rápido de Apresentação — AutoPredict AI 🚗🤖

---

## 🎧 Roteiro em Áudio — Leia em voz alta ou converta para MP3

> Cole este texto num conversor texto-para-voz (ex: ttsmp3.com, Google TTS, ou o leitor do seu celular) e escute no caminho.

---

O projeto se chama AutoPredict AI. É uma plataforma de inteligência artificial para diagnóstico e manutenção preditiva de veículos.

O problema que ele resolve é simples: empresas com frotas de veículos sofrem com quebras inesperadas. Um caminhão que quebra na estrada gera custo de reboque, conserto emergencial, atraso na entrega e cliente insatisfeito. Os dados para prever essa falha existiam, nos sensores do veículo, mas ninguém conseguia interpretar tudo isso em tempo real.

A solução é um chatbot especialista em mecânica, que estudou onze mil e cem casos reais de falhas e manutenção, e responde perguntas em português, rodando cem por cento no servidor local, sem mandar nada pra nuvem e sem custo de API.

A tecnologia principal se chama RAG, que significa Retrieval-Augmented Generation. Funciona assim: você faz uma pergunta, o sistema transforma essa pergunta em números, busca os quinze casos mais parecidos numa base de dados vetorial chamada Milvus, e manda esses casos reais para o modelo de linguagem responder. Isso garante que a resposta é baseada em dados reais, não em invenção do modelo.

Uma boa analogia é a de um bibliotecário especializado. A biblioteca são os três datasets do Kaggle com sessenta e um mil registros. O fichário de índices é o Milvus, que permite buscar por significado e não só por palavra exata. E o bibliotecário é o modelo de linguagem llama três ponto dois, rodando localmente pelo Ollama.

O pipeline de dados tem três camadas. A camada Bronze lê os arquivos CSV e salva no MinIO, que é um data lake local. A camada Silver limpa e normaliza os dados. E a camada Gold gera os embeddings e indexa tudo no Milvus.

Além do chatbot, o sistema também treina nove modelos de machine learning, sendo três algoritmos por dataset. Os algoritmos são Regressão Logística, Random Forest e XGBoost. O melhor resultado foi o XGBoost com F1 igual a um ponto zero no dataset de manutenção. Todos os experimentos ficam registrados no MLflow.

A stack tecnológica é: FastAPI no backend, Gradio no frontend, Milvus como banco vetorial, MinIO como data lake, Ollama com llama três ponto dois como modelo de linguagem, MLflow para rastrear experimentos, PostgreSQL para metadados, e tudo orquestrado com Docker Compose em um único comando.

Na demo, basta acessar localhost na porta sete oito seis zero, digitar uma pergunta como "quais os sinais de que o freio precisa de manutenção", e ativar a opção de mostrar os documentos usados como contexto para ver as fontes reais sendo usadas na resposta.

Para encerrar: o sistema está rodando com onze mil e cem documentos indexados, nove modelos treinados, e zero custo de API. Tudo local, tudo funcional.

---

## 1. O Problema (30 segundos)

> "Imagine que você tem uma frota de caminhões. Um deles quebra na estrada às 3h da manhã. Custo: reboque, conserto emergencial, atraso na entrega, cliente insatisfeito. Isso podia ter sido evitado."

O problema é: **os dados para prever essa falha existiam** — nos sensores do veículo — mas ninguém conseguia interpretar tudo isso em tempo real.

---

## 2. A Solução em Uma Frase

> "Um chatbot especialista em mecânica, que estudou 11.100 casos reais de falhas e manutenção, e responde suas perguntas em português, rodando 100% no seu servidor — sem mandar nada pra nuvem."

---

## 3. Como Funciona — Analogia do Bibliotecário

Pensa no sistema como um **bibliotecário muito especializado**:

| Peça | Analogia | Tecnologia |
|---|---|---|
| Biblioteca | 3 bases de dados reais (61.100 registros) | CSVs do Kaggle |
| Fichário de índices | Busca por *significado*, não por palavra | Milvus (banco vetorial) |
| Bibliotecário | Lê os casos relevantes e formula a resposta | LLM llama3.2:3b (Ollama) |

**Fluxo de uma pergunta:**

1. Você pergunta: *"O que causa superaquecimento do motor?"*
2. O sistema transforma a pergunta em números (embedding)
3. Busca os 15 casos mais parecidos no Milvus
4. Manda esses casos reais pro LLM junto com a pergunta
5. O LLM responde baseado naqueles dados — não em "alucinação"

Isso se chama **RAG (Retrieval-Augmented Generation)**.

---

## 4. Arquitetura em 3 Camadas

```
Usuário digita pergunta
       ↓
   Frontend (Gradio :7860)
       ↓
   API REST (FastAPI :8000)
       ↓
  ┌────┴────┐
Busca     Gera resposta
Milvus    Ollama (LLM local)
```

---

## 5. Pipeline de Dados — Analogia da Refinaria

| Camada | Analogia | O que faz |
|---|---|---|
| **Bronze** | Minério bruto | Lê os CSVs e salva no MinIO |
| **Silver** | Metal purificado | Limpa, normaliza, remove nulos |
| **Gold** | Produto final | Gera embeddings e indexa no Milvus |

> Resultado: **11.100 documentos** indexados prontos para busca semântica.

---

## 6. Modelos de ML (Bônus)

Além do RAG, foram treinados **9 modelos preditivos** (3 algoritmos × 3 datasets):

| Algoritmo | Maintenance (F1) | Engine Fault (F1) |
|---|---|---|
| Logistic Regression | 0.8959 | — |
| Random Forest | 0.9796 | — |
| **XGBoost** | **1.0000** | — |

Todos os experimentos estão registrados no **MLflow** em http://localhost:5001.

---

## 7. Stack Tecnológica

| Papel | Tecnologia |
|---|---|
| LLM local (sem custo de API) | Ollama + llama3.2:3b |
| Banco vetorial (busca semântica) | Milvus |
| Data Lake | MinIO (S3 local) |
| Backend / API REST | FastAPI |
| Frontend / Chat | Gradio |
| ML Tracking | MLflow |
| Infraestrutura | Docker Compose (1 comando sobe tudo) |

---

## 8. Demo — Passo a Passo

1. Acesse **http://localhost:7860**
2. Digite uma das perguntas abaixo:
   - *"Quais os sinais de que o freio precisa de manutenção?"*
   - *"O que causa superaquecimento do motor?"*
   - *"Meu carro tem 80.000km, o que verificar?"*
3. Ative **"Mostrar documentos usados como contexto"** para mostrar as fontes reais ao vivo

---

## 9. Serviços Rodando

| Serviço | URL |
|---|---|
| Chatbot (Gradio) | http://localhost:7860 |
| API REST | http://localhost:8000/docs |
| MLflow (experimentos) | http://localhost:5001 |
| MinIO (data lake) | http://localhost:9001 |

---

## 10. Frase de Encerramento

> "O sistema está rodando agora, com 11.100 documentos indexados, 9 modelos treinados, e zero custo de API. Tudo local, tudo funcional."
