# ============================================================
# Rota de Models - Lista modelos LLM disponíveis
# ============================================================
# GET /models → Retorna modelos disponíveis (Ollama + Cloud)
# ============================================================

import logging
import requests
from fastapi import APIRouter
from src.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/models", tags=["Models"])
def get_available_models():
    """
    Retorna lista de modelos LLM disponíveis.
    
    Inclui:
    - Modelos Ollama locais (consultando API do Ollama)
    - Modelos OpenAI (se API key configurada)
    - Modelos Anthropic (se API key configurada)
    
    Útil para popular dropdown no frontend.
    """
    logger.info("[API] GET /models")
    
    settings = get_settings()
    models = {
        "ollama": [],
        "openai": [],
        "anthropic": [],
        "groq": [],
        "default": settings.ollama_model,
    }
    
    # ── Busca modelos Ollama (local) ──────────────────────────────
    try:
        response = requests.get(f"{settings.ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            ollama_models = response.json().get("models", [])
            models["ollama"] = [
                {
                    "name": m["name"],
                    "display_name": f"{m['name']} (Ollama - Local)",
                    "provider": "ollama",
                    "local": True,
                }
                for m in ollama_models
            ]
            logger.info(f"[Models] Encontrados {len(models['ollama'])} modelos Ollama")
    except Exception as e:
        logger.warning(f"[Models] Não foi possível conectar ao Ollama: {e}")
        # Fallback: retorna apenas o modelo padrão
        models["ollama"] = [
            {
                "name": settings.ollama_model,
                "display_name": f"{settings.ollama_model} (Ollama - Local)",
                "provider": "ollama",
                "local": True,
            }
        ]
    
    # ── Modelos OpenAI (cloud) ────────────────────────────────────
    # Sempre mostra, mas indica se precisa configurar API key
    has_openai_key = bool(settings.openai_api_key)
    models["openai"] = [
        {
            "name": "gpt-4o", 
            "display_name": f"GPT-4 Omni (OpenAI){' ⚠️ Configure API Key' if not has_openai_key else ''}", 
            "provider": "openai", 
            "local": False,
            "requires_key": not has_openai_key,
        },
        {
            "name": "gpt-4", 
            "display_name": f"GPT-4 (OpenAI){' ⚠️ Configure API Key' if not has_openai_key else ''}", 
            "provider": "openai", 
            "local": False,
            "requires_key": not has_openai_key,
        },
        {
            "name": "gpt-3.5-turbo", 
            "display_name": f"GPT-3.5 Turbo (OpenAI){' ⚠️ Configure API Key' if not has_openai_key else ''}", 
            "provider": "openai", 
            "local": False,
            "requires_key": not has_openai_key,
        },
    ]
    
    # ── Modelos Groq (cloud GRATUITO) ─────────────────────────────
    # Sempre mostra, mas indica se precisa configurar API key
    has_groq_key = bool(settings.groq_api_key)
    models["groq"] = [
        {
            "name": "llama-3.1-70b-versatile", 
            "display_name": f"Llama 3.1 70B (Groq - Grátis){' ⚠️ Configure API Key' if not has_groq_key else ' ✨'}", 
            "provider": "groq", 
            "local": False,
            "requires_key": not has_groq_key,
        },
        {
            "name": "mixtral-8x7b-32768", 
            "display_name": f"Mixtral 8x7B (Groq - Grátis){' ⚠️ Configure API Key' if not has_groq_key else ' ✨'}", 
            "provider": "groq", 
            "local": False,
            "requires_key": not has_groq_key,
        },
        {
            "name": "gemma-7b-it", 
            "display_name": f"Gemma 7B (Groq - Grátis){' ⚠️ Configure API Key' if not has_groq_key else ' ✨'}", 
            "provider": "groq", 
            "local": False,
            "requires_key": not has_groq_key,
        },
    ]
    
    # ── Modelos Anthropic (cloud) ─────────────────────────────────
    # Sempre mostra, mas indica se precisa configurar API key
    has_anthropic_key = bool(settings.anthropic_api_key)
    models["anthropic"] = [
        {
            "name": "claude-3-opus-20240229", 
            "display_name": f"Claude 3 Opus (Anthropic){' ⚠️ Configure API Key' if not has_anthropic_key else ''}", 
            "provider": "anthropic", 
            "local": False,
            "requires_key": not has_anthropic_key,
        },
        {
            "name": "claude-3-sonnet-20240229", 
            "display_name": f"Claude 3 Sonnet (Anthropic){' ⚠️ Configure API Key' if not has_anthropic_key else ''}", 
            "provider": "anthropic", 
            "local": False,
            "requires_key": not has_anthropic_key,
        },
        {
            "name": "claude-3-haiku-20240307", 
            "display_name": f"Claude 3 Haiku (Anthropic){' ⚠️ Configure API Key' if not has_anthropic_key else ''}", 
            "provider": "anthropic", 
            "local": False,
            "requires_key": not has_anthropic_key,
        },
    ]
    
    # ── Retorna lista unificada ───────────────────────────────────
    all_models = models["ollama"] + models["groq"] + models["openai"] + models["anthropic"]
    
    return {
        "models": all_models,
        "default_model": settings.ollama_model,
        "total_available": len(all_models),
        "by_provider": {
            "ollama": len(models["ollama"]),
            "groq": len(models["groq"]),
            "openai": len(models["openai"]),
            "anthropic": len(models["anthropic"]),
        },
    }
