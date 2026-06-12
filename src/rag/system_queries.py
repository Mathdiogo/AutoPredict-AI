# ============================================================
# Perguntas sobre o sistema — respostas diretas (sem RAG veicular)
# ============================================================
# Detecta perguntas meta (quem é você, modelos ML, pré-processamento)
# e retorna respostas fixas. Perguntas sobre carros seguem o RAG normal.
# ============================================================

import re
import unicodedata


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text)


# Ordem importa: padrões mais específicos primeiro
_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    (
        "ml_models",
        [
            r"modelos? (foram )?treinados?",
            r"modelos? de ml\b",
            r"modelos? de machine learning",
            r"quais (algoritmos|modelos).*trein",
            r"metricas? (foram )?(usadas|utilizadas)",
            r"modelos?.*metricas?",
            r"metricas?.*modelos?",
            r"quais metricas",
        ],
    ),
    (
        "preprocessing",
        [
            r"pre[- ]?processamento",
            r"preprocessamento",
            r"limpeza (de |dos )?dados",
            r"transformacoes? (nos|dos|de|aplicadas)",
            r"camada silver",
            r"como os dados (foram )?(tratados|limpos|processados)",
            r"padronizacao dos dados",
            r"normalizacao dos dados",
        ],
    ),
    (
        "identity",
        [
            r"^quem (e|eh) (voce|vc|o sistema|autopredict)",
            r"^o que (e|eh) (voce|vc|autopredict)",
            r"^apresente(-se)?",
            r"^qual (e|eh) seu nome",
            r"^quem (e|eh) autopredict",
            r"^me (fale|conte) sobre (voce|vc|o sistema|autopredict)",
            r"^o que (e|eh) autopredict ai",
        ],
    ),
]

# Indícios de pergunta veicular — não tratar como meta mesmo com palavras ambíguas
_VEHICLE_HINTS = [
    r"\bcarro\b", r"\bveiculo\b", r"\bmotor\b", r"\bpneu", r"\bfreio",
    r"\boleo\b", r"\bkm\b", r"\bquilometragem\b", r"\bsensor",
    r"\bmanutencao\b", r"\bfalha\b", r"\bdiagnostico automotivo\b",
]


def detect_system_query(question: str) -> str | None:
    """
    Retorna o tipo de pergunta meta ou None se for consulta veicular/normal.
    """
    normalized = _normalize(question)

    if any(re.search(p, normalized) for p in _VEHICLE_HINTS):
        # Ex.: "quais modelos de carro foram treinados" → RAG veicular
        if not any(
            re.search(p, normalized)
            for p in [
                r"modelos? (foram )?treinados?",
                r"modelos? de ml",
                r"metricas?",
                r"pre[- ]?processamento",
            ]
        ):
            return None

    for intent, patterns in _INTENT_PATTERNS:
        if any(re.search(p, normalized) for p in patterns):
            return intent

    return None


def get_system_response(intent: str) -> str:
    """Resposta direta para perguntas meta — sem contexto veicular."""
    responses = {
        "identity": """Olá! Sou o **AutoPredict AI**, um assistente especializado em **diagnóstico e manutenção preditiva de veículos**.

Combino:
- **RAG** (busca semântica em 3 bases de dados automotivos + resposta com LLM)
- **Machine Learning** (Logistic Regression, Random Forest e XGBoost) para predição de falhas
- **Governança de dados** no padrão Medallion (Bronze → Silver → Gold), com rastreio no MLflow

Posso responder perguntas sobre manutenção, sensores e falhas de motor com base nos dados indexados. Também posso explicar os modelos treinados e o pré-processamento dos dados, se você perguntar.""",

        "ml_models": """Foram treinados **3 algoritmos** para **cada um dos 3 datasets**, totalizando **9 modelos** registrados no MLflow:

**Algoritmos:** Logistic Regression, Random Forest e XGBoost

**Métricas registradas:** Accuracy, F1-score (weighted) e AUC-ROC (+ classification report como artefato)

**Experimentos MLflow:**
1. **AutoPredict-Maintenance** (vehicle_maintenance) — melhor: XGBoost (Acc=1.00, F1=1.00, AUC=1.00)
2. **AutoPredict-Predictive** (car_predictive) — melhor: Random Forest (F1=0.56, AUC=0.56)
3. **AutoPredict-EngineFault** (engine_fault) — modelos registrados com as mesmas métricas

Consulte detalhes e comparações em: http://localhost:5001""",

        "preprocessing": """O pré-processamento segue o pipeline **Medallion** (camada **Silver**):

**1. Bronze → ingestão bruta**
- CSVs originais salvos no MinIO (`bronze/`) sem alteração

**2. Silver → limpeza e padronização**
- Remoção de linhas totalmente vazias e **duplicatas**
- **Normalização de nomes de colunas** (minúsculas, sem caracteres especiais, espaços → `_`)
- **Conversão de tipos** (colunas numéricas de sensores)
- Remoção de **valores impossíveis** (ex.: temperatura fora de -50°C a 300°C)
- Preenchimento de nulos numéricos com **mediana**
- Flags binárias (`failure_flag`, `anomaly_indication`) padronizadas em 0/1
- Limite de **5.000 linhas por dataset** na camada Gold (para indexação)

**3. Gold → embeddings**
- Texto descritivo por linha → vetores com `paraphrase-multilingual-MiniLM-L12-v2` → indexação no Milvus

**Governança documentada no MinIO:** bucket `governance/` (pasta `governance/` com bronze, silver, gold e políticas)

Documentação local: `docs/governance/silver_layer.md`""",
    }
    return responses.get(intent, "")
