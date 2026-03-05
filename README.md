# AutoPredict AI 🚗🤖

<div align="center">

**Plataforma RAG para Diagnóstico Automotivo Preditivo**

![Status](https://img.shields.io/badge/Status-Sprint%201-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![AI](https://img.shields.io/badge/AI-RAG-orange)

</div>

---

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Domínio](#domínio)
- [Problema de Negócio](#problema-de-negócio)
- [Solução Proposta](#solução-proposta)
- [Arquitetura](#arquitetura)
- [Datasets](#datasets)
- [Requisitos](#requisitos)
- [Metodologia Scrum](#metodologia-scrum)
- [Product Backlog](#product-backlog)
- [Tecnologias](#tecnologias)
- [Roadmap](#roadmap)

---

## 🎯 Sobre o Projeto

A **AutoPredict AI** é uma plataforma de inteligência artificial especializada em **manutenção preditiva de veículos**, utilizando técnicas avançadas de **Retrieval-Augmented Generation (RAG)** para auxiliar no diagnóstico e prevenção de falhas automotivas.

### Objetivos

- 🔧 Reduzir custos de manutenção
- 🛡️ Aumentar a segurança dos veículos
- ⚡ Evitar paradas inesperadas
- 📊 Melhorar a gestão de frotas
- 🤖 Auxiliar técnicos e engenheiros no diagnóstico

---

## 🏭 Domínio

**Manutenção Preditiva de Veículos**

O projeto opera no domínio de manutenção preditiva automotiva, utilizando:

- 📡 Dados de sensores automotivos
- 📝 Histórico de manutenção
- ⚠️ Registros de falhas
- 🔍 Análise de padrões

A aplicação de IA permite identificar padrões que indicam possíveis problemas mecânicos ou eletrônicos **antes** que ocorram falhas críticas.

---

## 💼 Problema de Negócio

### Desafios Enfrentados

Empresas que operam frotas de veículos (transporte, logística, mobilidade) enfrentam:

- ❌ Falhas inesperadas em veículos
- 💰 Custos elevados de manutenção
- 📊 Dificuldade de analisar grandes volumes de dados de sensores
- 🔧 Diagnóstico lento de problemas mecânicos/eletrônicos

### Consequências

- ⏱️ Atrasos operacionais
- 📈 Aumento de custos
- ⚠️ Riscos de segurança
- 🚫 Paradas não planejadas

---

## 💡 Solução Proposta

Desenvolvimento de uma **plataforma baseada em IA** que permite:

✅ Consultar dados de veículos de forma inteligente  
✅ Identificar possíveis causas de falhas  
✅ Auxiliar engenheiros e técnicos na tomada de decisão  
✅ Prever necessidade de manutenção  
✅ Analisar padrões de falhas históricos  

---

## 🏗️ Arquitetura

A solução é composta por três camadas principais:

### 📦 Camada de Dados

- **Data Lake**: MinIO (armazenamento de objetos)
- **Banco Relacional**: PostgreSQL (metadados)
- **Banco Vetorial**: Milvus (embeddings)
- **Padrão Medallion**: Bronze → Silver → Gold

### 🧠 Camada de IA

- **LLM Local**: Ollama
- **Pipeline RAG**: Retrieval-Augmented Generation
- **Embeddings**: Modelos de vetorização

### 🌐 Camada de Aplicação

- **API REST**: FastAPI
- **Interface**: Gradio
- **Documentação**: Swagger/OpenAPI

### 🐳 Infraestrutura

- Docker & Docker Compose
- Makefile para automação
- Ambiente containerizado

```
┌─────────────────────────────────────────────────┐
│              Interface (Gradio)                 │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│              API REST (FastAPI)                 │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│           Pipeline RAG + LLM (Ollama)           │
└────┬───────────┬───────────────┬────────────────┘
     │           │               │
┌────▼────┐ ┌───▼─────┐   ┌─────▼──────┐
│ MinIO   │ │ Milvus  │   │ PostgreSQL │
│(Data    │ │(Vetores)│   │ (Metadata) │
│ Lake)   │ │         │   │            │
└─────────┘ └─────────┘   └────────────┘
```

---

## 📊 Datasets

Para simular os dados da AutoPredict AI, serão utilizados datasets públicos relacionados a manutenção e diagnóstico automotivo:

### Datasets Selecionados

1. **[Vehicle Maintenance Data](https://www.kaggle.com/datasets/chavindudulaj/vehicle-maintenance-data)**
   - Registros de manutenção preventiva e corretiva
   - Histórico de peças substituídas

2. **[Car Predictive Maintenance Data](https://www.kaggle.com/datasets/pragyanaianddsschool/car-predictive-maintenance-data)**
   - Dados de sensores em tempo real
   - Padrões de falhas

3. **[Engine Fault Detection Data](https://www.kaggle.com/datasets/ziya07/engine-fault-detection-data)**
   - Falhas de motor
   - Códigos de diagnóstico (DTC)

### Informações Contidas

- 🌡️ Temperatura do motor
- 🛢️ Pressão do óleo
- 📳 Vibração
- ⚠️ Falhas detectadas
- 📅 Histórico de manutenção
- 🔧 Componentes afetados

---

## 📝 Requisitos

### Requisitos Funcionais

| ID | Descrição |
|----|-----------|
| **RF1** | O sistema deve permitir consultas sobre dados de manutenção de veículos |
| **RF2** | O sistema deve recuperar informações relevantes a partir de um banco de dados de documentos e registros |
| **RF3** | O sistema deve gerar respostas usando um modelo de linguagem (LLM) |
| **RF4** | O sistema deve permitir consulta via interface web |
| **RF5** | O sistema deve fornecer informações relacionadas a falhas e possíveis diagnósticos |

### Requisitos Não Funcionais

| ID | Descrição |
|----|-----------|
| **RNF1** | O sistema deve ser executado em ambiente containerizado utilizando Docker |
| **RNF2** | A arquitetura deve seguir o padrão de governança de dados Medallion (Bronze, Silver, Gold) |
| **RNF3** | O sistema deve possuir uma API REST para acesso às funcionalidades |
| **RNF4** | Os dados devem ser armazenados em um Data Lake |
| **RNF5** | O sistema deve possuir documentação técnica |

---

## 🏃 Metodologia Scrum

O projeto segue a metodologia **Scrum** com sprints de 2 semanas.

### Papéis Definidos

| Papel | Responsabilidade |
|-------|------------------|
| **Product Owner** | Definir requisitos do produto e priorizar o backlog |
| **Scrum Master** | Garantir o processo Scrum e remover impedimentos |
| **Development Team** | Desenvolvimento técnico do sistema |

### Distribuição de Papéis

| Papel | Responsável |
|-------|-------------|
| Product Owner | Integrante do grupo |
| Scrum Master | Integrante do grupo |
| Desenvolvedor Backend | Integrante do grupo |
| Desenvolvedor Dados/IA | Integrante do grupo |

---

## 📋 Product Backlog

### Sprint 1 - Definição do Produto ✅

| ID | Item | Prioridade | Status |
|----|------|------------|--------|
| 1 | Definir domínio do projeto | Alta | ✅ Concluído |
| 2 | Definir empresa fictícia | Alta | ✅ Concluído |
| 3 | Definir problema de negócio | Alta | ✅ Concluído |
| 4 | Selecionar datasets públicos | Alta | ✅ Concluído |
| 5 | Definir arquitetura inicial do sistema | Alta | ✅ Concluído |
| 6 | Configurar ambiente Docker | Média | 🔄 Em progresso |
| 7 | Criar Data Lake no MinIO | Média | 📅 Planejado |
| 8 | Criar banco PostgreSQL para metadados | Média | 📅 Planejado |
| 9 | Implementar pipeline de ingestão de dados | Média | 📅 Planejado |
| 10 | Definir pipeline RAG | Média | 📅 Planejado |

---

## 🛠️ Tecnologias

### Backend & API
![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)

### Inteligência Artificial
![Ollama](https://img.shields.io/badge/Ollama-LLM-orange)
![RAG](https://img.shields.io/badge/RAG-Pipeline-purple)

### Dados
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?logo=postgresql&logoColor=white)
![MinIO](https://img.shields.io/badge/MinIO-C72E49?logo=minio&logoColor=white)
![Milvus](https://img.shields.io/badge/Milvus-Vector%20DB-blue)

### Interface
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?logo=gradio&logoColor=white)

### Infraestrutura
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Docker%20Compose-2496ED?logo=docker&logoColor=white)

---

## 🗺️ Roadmap

### Sprint 1 - Definição do Produto (Atual) ✅
- [x] Escolha do domínio
- [x] Definição da empresa fictícia
- [x] Definição do problema de negócio
- [x] Levantamento de requisitos
- [x] Definição dos papéis Scrum
- [x] Product Backlog inicial

### Sprint 2 - Infraestrutura Base 📅
- [ ] Configuração do Docker Compose
- [ ] Setup do MinIO (Data Lake)
- [ ] Setup do PostgreSQL
- [ ] Setup do Milvus
- [ ] Documentação de setup

### Sprint 3 - Ingestão de Dados 📅
- [ ] Pipeline de ingestão (Bronze)
- [ ] Limpeza de dados (Silver)
- [ ] Dados processados (Gold)
- [ ] Testes de pipeline

### Sprint 4 - Pipeline RAG 📅
- [ ] Implementação do RAG
- [ ] Integração com LLM (Ollama)
- [ ] Geração de embeddings
- [ ] Testes de recuperação

### Sprints 5-12 📅
- [ ] API REST
- [ ] Interface Gradio
- [ ] Testes e validação
- [ ] Documentação final
- [ ] Deploy

---

## 📄 Entregáveis da Sprint 1

### ✅ Documentação
- [x] README.md completo
- [x] Definição do domínio
- [x] Descrição da empresa fictícia
- [x] Problema de negócio identificado
- [x] Requisitos levantados

### ✅ Planejamento
- [x] Papéis Scrum definidos
- [x] Product Backlog inicial
- [x] Arquitetura conceitual
- [x] Datasets selecionados

---

## 👥 Equipe

*[Adicione aqui os nomes dos integrantes do grupo]*

---

## 📞 Contato

Para mais informações sobre o projeto AutoPredict AI:

- 📧 Email: [matheusponte2010@hotmail.com]
- 🐙 GitHub: [github.com/Mathdiogo/AutoPredict-AI]

---

## 📜 Licença

Este projeto é desenvolvido para fins acadêmicos.

---

<div align="center">

**AutoPredict AI** - Prevendo o futuro da manutenção automotiva 🚗✨

*Desenvolvido com ❤️ por [Nome da Equipe]*

</div>
