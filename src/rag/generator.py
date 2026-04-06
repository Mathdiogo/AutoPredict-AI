# ============================================================
# RAG Generator - Geração de Resposta com LLM (Ollama)
# ============================================================
# Esta é a parte do "G" no RAG (Generation).
#
# O que acontece aqui:
#   1. Recebe a pergunta + documentos recuperados do Retriever
#   2. Monta um PROMPT estruturado com todo o contexto
#   3. Envia para o Ollama (LLM local)
#   4. Retorna a resposta gerada
#
# CONCEITO - O Prompt:
#   O prompt é a instrução que mandamos para o LLM.
#   Um bom prompt tem:
#     - Papel do assistente ("Você é um especialista em...")
#     - Contexto (os documentos recuperados)
#     - A pergunta do usuário
#     - Instrução de comportamento ("Responda em português...")
#
# O LLM usa o contexto para dar uma resposta fundamentada
# nos dados reais em vez de inventar (alucinar).
# ============================================================

import logging
import requests
from dataclasses import dataclass
from src.rag.retriever import RetrievedDocument
from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GeneratorResponse:
    """Resposta completa do pipeline RAG."""
    answer: str                         # Resposta gerada pelo LLM
    sources: list[RetrievedDocument]    # Documentos usados como contexto
    query: str                          # Pergunta original
    model_used: str                     # Modelo Ollama usado


def _build_prompt(query: str, documents: list[RetrievedDocument]) -> str:
    """
    Constrói o prompt que será enviado ao LLM.

    ESTRUTURA:
      [SISTEMA]  - Define o papel do assistente
      [CONTEXTO] - Os documentos recuperados dos 3 datasets
      [PERGUNTA] - A pergunta do usuário
    """
    # ── Agrupa documentos por fonte para clareza ──────────────────
    docs_by_source: dict[str, list[RetrievedDocument]] = {}
    for doc in documents:
        label = doc.source_label
        if label not in docs_by_source:
            docs_by_source[label] = []
        docs_by_source[label].append(doc)

    # ── Monta o bloco de contexto ─────────────────────────────────
    context_blocks = []
    for source_label, docs in docs_by_source.items():
        context_blocks.append(f"### {source_label}")
        for i, doc in enumerate(docs, 1):
            # Garante que o texto não seja muito longo (evita estourar o contexto do LLM)
            text = doc.text[:800] if len(doc.text) > 800 else doc.text
            context_blocks.append(f"{i}. {text}")
        context_blocks.append("")  # Linha em branco entre seções

    context = "\n".join(context_blocks)

    # ── Prompt final ──────────────────────────────────────────────
    prompt = f"""Você é AutoPredict AI, especialista em diagnóstico e manutenção preditiva de veículos.
Sua resposta deve ser técnica, estruturada e baseada nos dados reais fornecidos.

Fontes de dados disponíveis:
\u2022 \U0001f4cb Histórico de Manutenção \u2014 registros de serviços, peças e quilometragem
\u2022 \U0001f4ca Sensores Preditivos \u2014 temperatura do motor, pressão dos pneus, espessura do freio
\u2022 \u26a0\ufe0f Diagnóstico de Falhas \u2014 vibração, temperatura de exaustão, pressão de admissão

--- EXEMPLOS DE COMO RESPONDER ---

Pergunta: "Meu motor está superaquecendo, o que pode ser?"
Resposta: Com base nos dados de sensores (\U0001f4ca), temperatura acima de 100°C indica superaquecimento.
As causas mais comuns encontradas no histórico (\U0001f4cb) são: termostato defeituoso, falta de
refrigerante e bomba d'água com falha. Recomendação: verificar nível do radiador imediatamente
e agendar revisão do sistema de arrefecimento.

Pergunta: "Com que frequência devo trocar o óleo?"
Resposta: O histórico de manutenção (\U0001f4cb) indica troca a cada 5.000-10.000km dependendo do tipo
de combustível. Veículos a diesel apresentam maior frequência de manutenção nos dados analisados.

--- FIM DOS EXEMPLOS ---

DADOS REAIS DO SISTEMA PARA ESTA CONSULTA:
{context}

PERGUNTA: {query}

INSTRUÇÕES PARA RESPOSTA:
- Responda em português brasileiro
- Estruture com causas, indicadores e recomendações práticas
- Cite as fontes com os ícones (\U0001f4cb \U0001f4ca \u26a0\ufe0f)
- Se dados forem insuficientes, indique o que seria necessário verificar
- Seja direto e técnico; evite frases genéricas sem embasamento nos dados

RESPOSTA:"""

    return prompt


class Generator:
    """
    Gerencia a geração de respostas via API do Ollama.
    """

    def __init__(self):
        self.settings = get_settings()

    def generate(self, query: str, documents: list[RetrievedDocument]) -> GeneratorResponse:
        """
        Gera uma resposta baseada na pergunta e nos documentos recuperados.

        Args:
            query: Pergunta do usuário
            documents: Documentos relevantes retornados pelo Retriever

        Returns:
            GeneratorResponse com a resposta e metadados
        """
        prompt = _build_prompt(query, documents)

        logger.info(f"[Generator] Enviando para Ollama (modelo: {self.settings.ollama_model})")
        logger.debug(f"[Generator] Prompt tamanho: {len(prompt)} chars")

        try:
            response = requests.post(
                f"{self.settings.ollama_url}/api/generate",
                json={
                    "model": self.settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,        # False = aguarda resposta completa
                    "options": {
                        "temperature": 0.2,  # Mais determinístico = respostas mais consistentes
                        "top_p": 0.9,        # Controla diversidade dos tokens
                        "num_predict": 800,  # Máximo de tokens (limitado para evitar timeout no CPU)
                    },
                },
                timeout=180,  # 3 minutos de timeout (llama3.2:3b gera ~100 tok/s no CPU)
            )
            response.raise_for_status()

            result = response.json()
            answer = result.get("response", "Não foi possível gerar uma resposta.")

            logger.info(f"[Generator] Resposta gerada ({len(answer)} chars)")

            return GeneratorResponse(
                answer=answer,
                sources=documents,
                query=query,
                model_used=self.settings.ollama_model,
            )

        except requests.exceptions.Timeout:
            error_msg = (
                "⚠️ O modelo demorou muito para responder. "
                "Isso pode acontecer na primeira resposta (modelo sendo carregado) "
                "ou se o servidor estiver sobrecarregado. Tente novamente."
            )
            logger.error("[Generator] Timeout ao chamar Ollama")
            return GeneratorResponse(
                answer=error_msg,
                sources=documents,
                query=query,
                model_used=self.settings.ollama_model,
            )

        except requests.exceptions.ConnectionError:
            error_msg = (
                "⚠️ Não foi possível conectar ao Ollama. "
                "Verifique se o serviço está rodando com: docker compose ps"
            )
            logger.error("[Generator] Erro de conexão com Ollama")
            return GeneratorResponse(
                answer=error_msg,
                sources=documents,
                query=query,
                model_used=self.settings.ollama_model,
            )

        except Exception as e:
            logger.error(f"[Generator] Erro inesperado: {e}")
            return GeneratorResponse(
                answer=f"Erro ao gerar resposta: {str(e)}",
                sources=documents,
                query=query,
                model_used=self.settings.ollama_model,
            )

    def stream_generate(self, query: str, documents: list[RetrievedDocument]):
        """
        Versão com streaming: yields tokens conforme são gerados.
        Usado para mostrar a resposta em tempo real no Gradio.

        Usage:
            for token in generator.stream_generate(query, docs):
                print(token, end="", flush=True)
        """
        prompt = _build_prompt(query, documents)

        try:
            with requests.post(
                f"{self.settings.ollama_url}/api/generate",
                json={
                    "model": self.settings.ollama_model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_predict": 800,
                    },
                },
                stream=True,
                timeout=180,
            ) as response:
                response.raise_for_status()
                import json
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break

        except Exception as e:
            logger.error(f"[Generator] Erro no streaming: {e}")
            yield f"\n[Erro: {str(e)}]"
