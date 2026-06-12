# ============================================================
# RAG Generator - Geração de Resposta com LLM Multi-Provider
# ============================================================
# Esta é a parte do "G" no RAG (Generation).
#
# SUPORTA 4 PROVIDERS:
#   1. Ollama (Local) - llama3.2:3b, mistral, qwen2.5:3b, etc.
#   2. OpenAI (Cloud) - gpt-4, gpt-3.5-turbo, gpt-4o, etc.
#   3. Anthropic (Cloud) - claude-3-opus, claude-3-sonnet, etc.
#   4. Groq (Cloud GRATUITO) - llama-3.1-70b, mixtral-8x7b, gemma-7b
#
# O que acontece aqui:
#   1. Recebe a pergunta + documentos recuperados do Retriever
#   2. Monta um PROMPT estruturado com todo o contexto
#   3. Detecta qual provider usar com base no modelo
#   4. Envia para a API apropriada
#   5. Retorna a resposta gerada + métricas de governança
# ============================================================

import logging
import requests
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from src.rag.retriever import RetrievedDocument
from src.config import get_settings

logger = logging.getLogger(__name__)

# Modelos Groq descontinuados → substitutos atuais
GROQ_MODEL_ALIASES = {
    "llama-3.1-70b-versatile": "llama-3.3-70b-versatile",
    "llama-3.1-70b-specdec": "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768": "llama-3.3-70b-versatile",
    "gemma-7b-it": "llama-3.1-8b-instant",
    "gemma2-9b-it": "llama-3.1-8b-instant",
}


@dataclass
class GeneratorResponse:
    """Resposta completa do pipeline RAG com métricas de governança."""
    answer: str                         # Resposta gerada pelo LLM
    sources: list[RetrievedDocument]    # Documentos usados como contexto
    query: str                          # Pergunta original
    model_used: str                     # Modelo LLM usado
    provider: str = ""                  # Provider (ollama, openai, etc)
    inference_time: float = 0.0         # Tempo de inferência em segundos
    tokens_used: int = 0                # Tokens estimados
    generation_params: dict = field(default_factory=dict)  # top_p, top_k, temperature


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

    # ── Prompt final (consultas veiculares — usa apenas o contexto recuperado) ──
    prompt = f"""Você é AutoPredict AI, especialista em diagnóstico e manutenção preditiva de veículos.

Fontes de dados no contexto abaixo:
• 📋 Histórico de Manutenção — serviços, peças e quilometragem
• 📊 Sensores Preditivos — temperatura, pressão dos pneus, espessura do freio
• ⚠️ Diagnóstico de Falhas — vibração, temperatura de exaustão, pressão de admissão

--- EXEMPLOS DE COMO RESPONDER ---

Pergunta: "Meu motor está superaquecendo, o que pode ser?"
Resposta: Com base nos dados de sensores (📊), temperatura acima de 100°C indica superaquecimento.
As causas mais comuns encontradas no histórico (📋) são: termostato defeituoso, falta de
refrigerante e bomba d'água com falha. Recomendação: verificar nível do radiador imediatamente
e agendar revisão do sistema de arrefecimento.

--- FIM DOS EXEMPLOS ---

DADOS RECUPERADOS PARA ESTA CONSULTA:
{context}

PERGUNTA: {query}

INSTRUÇÕES:
- Responda em português brasileiro
- Responda SOMENTE o que foi perguntado — não invente dados de veículos
- Use APENAS as informações do contexto acima; se insuficiente, diga o que falta
- Para perguntas sobre veículos: estruture com causas, indicadores e recomendações práticas
- Cite as fontes com os ícones (📋 📊 ⚠️) quando usar dados do contexto
- Seja direto; evite análises genéricas sem embasamento nos dados

RESPOSTA:"""

    return prompt


def _estimate_tokens(text: str) -> int:
    """
    Estima o número de tokens em um texto.
    Regra aproximada: 1 token ≈ 4 caracteres em português.
    """
    return max(1, len(text) // 4)


class Generator:
    """
    Gerencia a geração de respostas com suporte a múltiplos providers.
    """

    def __init__(self):
        self.settings = get_settings()

    def _detect_provider(self, model: Optional[str]) -> tuple[str, str]:
        """
        Detecta qual provider usar com base no nome do modelo.
        
        Returns:
            (provider, model_name)
            provider: 'ollama', 'openai', ou 'anthropic'
        """
        if model is None:
            return ("ollama", self.settings.ollama_model)
        
        model_lower = model.lower()
        
        # OpenAI models
        if any(x in model_lower for x in ["gpt", "chatgpt", "openai"]):
            return ("openai", model)
        
        # Anthropic models
        if any(x in model_lower for x in ["claude", "anthropic"]):
            return ("anthropic", model)
        
        # Groq models
        if any(x in model_lower for x in ["groq", "llama-3", "mixtral", "gemma"]):
            return ("groq", model)
        
        # Default: Ollama (local models)
        return ("ollama", model)

    def _generate_ollama(self, prompt: str, model: str, temperature: float = 0.2, 
                         top_p: float = 0.9, top_k: int = 40) -> tuple[str, int]:
        """
        Gera resposta usando Ollama (local).
        
        Returns:
            (resposta, tokens_usados)
        """
        try:
            response = requests.post(
                f"{self.settings.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "num_predict": self.settings.max_tokens_per_request,
                    },
                },
                timeout=180,
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "Não foi possível gerar uma resposta.")
            
            # Ollama retorna estatísticas de tokens
            eval_count = result.get("eval_count", _estimate_tokens(answer))
            prompt_eval_count = result.get("prompt_eval_count", _estimate_tokens(prompt))
            total_tokens = eval_count + prompt_eval_count
            
            return answer, total_tokens
        
        except requests.exceptions.Timeout:
            return ("⚠️ O modelo demorou muito para responder. "
                    "Isso pode acontecer na primeira resposta (modelo sendo carregado). Tente novamente.", 0)
        except requests.exceptions.ConnectionError:
            return ("⚠️ Não foi possível conectar ao Ollama. "
                    "Verifique se o serviço está rodando com: docker compose ps", 0)
        except Exception as e:
            logger.error(f"[Ollama] Erro: {e}")
            return f"Erro ao gerar resposta com Ollama: {str(e)}", 0

    def _generate_openai(self, prompt: str, model: str, temperature: float = 0.2,
                         top_p: float = 0.9) -> tuple[str, int]:
        """
        Gera resposta usando OpenAI API.
        
        Returns:
            (resposta, tokens_usados)
        """
        if not self.settings.openai_api_key:
            return "⚠️ OpenAI API key não configurada. Adicione OPENAI_API_KEY no .env", 0
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Você é AutoPredict AI, especialista em diagnóstico automotivo."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": self.settings.max_tokens_per_request,
                },
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            
            # OpenAI retorna usage com tokens exatos
            tokens = result.get("usage", {}).get("total_tokens", _estimate_tokens(prompt + answer))
            
            return answer, tokens
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return "⚠️ OpenAI API key inválida. Verifique OPENAI_API_KEY no .env", 0
            elif e.response.status_code == 429:
                return "⚠️ Limite de requisições da OpenAI atingido. Tente novamente em instantes.", 0
            return f"Erro na API OpenAI: {e.response.text}", 0
        except Exception as e:
            logger.error(f"[OpenAI] Erro: {e}")
            return f"Erro ao gerar resposta com OpenAI: {str(e)}", 0

    def _generate_anthropic(self, prompt: str, model: str, temperature: float = 0.2,
                            top_p: float = 0.9, top_k: int = 40) -> tuple[str, int]:
        """
        Gera resposta usando Anthropic API (Claude).
        
        Returns:
            (resposta, tokens_usados)
        """
        if not self.settings.anthropic_api_key:
            return "⚠️ Anthropic API key não configurada. Adicione ANTHROPIC_API_KEY no .env", 0
        
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.settings.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": self.settings.max_tokens_per_request,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                },
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            answer = result["content"][0]["text"]
            
            # Anthropic retorna usage com tokens
            tokens = result.get("usage", {}).get("input_tokens", 0) + result.get("usage", {}).get("output_tokens", 0)
            if tokens == 0:
                tokens = _estimate_tokens(prompt + answer)
                
            return answer, tokens
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return "⚠️ Anthropic API key inválida. Verifique ANTHROPIC_API_KEY no .env", 0
            return f"Erro na API Anthropic: {e.response.text}", 0
        except Exception as e:
            logger.error(f"[Anthropic] Erro: {e}")
            return f"Erro ao gerar resposta com Anthropic: {str(e)}", 0

    def _generate_groq(self, prompt: str, model: str, temperature: float = 0.2,
                       top_p: float = 0.9) -> tuple[str, int]:
        """
        Gera resposta usando Groq API (GRATUITO e extremamente rápido).
        
        Returns:
            (resposta, tokens_usados)
        """
        if not self.settings.groq_api_key:
            return "⚠️ Groq API key não configurada. Cadastre-se grátis em https://console.groq.com e adicione GROQ_API_KEY no .env", 0

        model = GROQ_MODEL_ALIASES.get(model, model)

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Você é um assistente especializado em diagnóstico automotivo."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": self.settings.max_tokens_per_request,
                },
                timeout=30,  # Groq é rápido!
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            
            # Groq retorna usage com tokens
            tokens = result.get("usage", {}).get("total_tokens", _estimate_tokens(prompt + answer))
            
            return answer, tokens
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return "⚠️ Groq API key inválida. Verifique GROQ_API_KEY no .env ou crie uma em https://console.groq.com", 0
            elif e.response.status_code == 429:
                return "⚠️ Limite de requisições do Groq atingido. Aguarde alguns segundos.", 0
            return f"Erro na API Groq: {e.response.text}", 0
        except Exception as e:
            logger.error(f"[Groq] Erro: {e}")
            return f"Erro ao gerar resposta com Groq: {str(e)}", 0

    def generate(self, query: str, documents: list[RetrievedDocument], 
                 model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None) -> GeneratorResponse:
        """
        Gera uma resposta baseada na pergunta e nos documentos recuperados.

        Args:
            query: Pergunta do usuário
            documents: Documentos relevantes retornados pelo Retriever
            model: Modelo a usar (None = padrão do config)
            temperature: Temperatura para geração (None = padrão do config)
            top_p: Parâmetro top_p (None = padrão do config)
            top_k: Parâmetro top_k (None = padrão do config)

        Returns:
            GeneratorResponse com a resposta e metadados
        """
        # Usa valores padrão se não fornecidos
        if temperature is None:
            temperature = self.settings.default_temperature
        if top_p is None:
            top_p = self.settings.default_top_p
        if top_k is None:
            top_k = self.settings.default_top_k
            
        prompt = _build_prompt(query, documents)
        provider, model_name = self._detect_provider(model)
        
        logger.info(f"[Generator] Provider: {provider}, Model: {model_name}")
        
        # Inicia medição de tempo
        start_time = time.time()
        
        # Gera com o provider apropriado
        if provider == "ollama":
            answer, tokens = self._generate_ollama(prompt, model_name, temperature, top_p, top_k)
        elif provider == "openai":
            answer, tokens = self._generate_openai(prompt, model_name, temperature, top_p)
        elif provider == "anthropic":
            answer, tokens = self._generate_anthropic(prompt, model_name, temperature, top_p, top_k)
        elif provider == "groq":
            answer, tokens = self._generate_groq(prompt, model_name, temperature, top_p)
        else:
            answer = f"Provider desconhecido: {provider}"
            tokens = 0

        # Calcula tempo de inferência
        inference_time = time.time() - start_time

        logger.info(f"[Generator] Resposta gerada ({len(answer)} chars, {tokens} tokens, {inference_time:.2f}s)")

        return GeneratorResponse(
            answer=answer,
            sources=documents,
            query=query,
            model_used=f"{provider}:{model_name}",
            provider=provider,
            inference_time=inference_time,
            tokens_used=tokens,
            generation_params={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )

    def stream_generate(self, query: str, documents: list[RetrievedDocument], model: Optional[str] = None):
        """
        Versão com streaming: yields tokens conforme são gerados.
        NOTA: Streaming só funciona com Ollama. OpenAI/Anthropic retornam resposta completa.

        Usage:
            for token in generator.stream_generate(query, docs):
                print(token, end="", flush=True)
        """
        prompt = _build_prompt(query, documents)
        provider, model_name = self._detect_provider(model)

        # Streaming só funciona com Ollama
        if provider != "ollama":
            logger.warning(f"[Generator] Streaming não suportado para {provider}, retornando resposta completa")
            answer = self.generate(query, documents, model).answer
            yield answer
            return

        try:
            with requests.post(
                f"{self.settings.ollama_url}/api/generate",
                json={
                    "model": model_name,
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
