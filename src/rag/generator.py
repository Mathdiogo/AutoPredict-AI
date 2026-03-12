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
    prompt = f"""Você é AutoPredict AI, um especialista em diagnóstico e manutenção preditiva de veículos.
Sua função é ajudar técnicos e engenheiros automotivos a identificar problemas e tomar decisões de manutenção.

Você tem acesso a dados reais de 3 fontes diferentes:
1. Histórico de manutenção de veículos
2. Dados de sensores preditivos (temperatura, pressão, vibração, etc.)
3. Banco de dados de falhas e diagnósticos técnicos

DADOS DISPONÍVEIS SOBRE O CASO:
{context}

PERGUNTA DO USUÁRIO:
{query}

INSTRUÇÕES:
- Responda em português brasileiro
- Base sua resposta PRINCIPALMENTE nos dados fornecidos acima
- Indique de qual fonte cada informação veio (Manutenção, Sensores ou Falhas)
- Se os dados sugerirem um problema específico, aponte a causa provável
- Dê recomendações práticas e acionáveis
- Se os dados forem insuficientes, diga claramente o que mais seria necessário saber
- Nunca invente dados que não foram fornecidos no contexto

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
                        "temperature": 0.3,  # Baixo = mais determinístico, menos criativo
                        "top_p": 0.9,        # Controla diversidade dos tokens
                        "num_predict": 1024, # Máximo de tokens na resposta
                    },
                },
                timeout=120,  # 2 minutos de timeout (modelos locais podem ser lentos)
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
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 1024,
                    },
                },
                stream=True,
                timeout=120,
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
