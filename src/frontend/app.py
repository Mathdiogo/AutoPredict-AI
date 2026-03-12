# ============================================================
# Gradio - Interface de Chat (Frontend)
# ============================================================
# Gradio cria interfaces web para modelos de IA com poucas linhas.
# Aqui criamos um chatbot que chama a nossa FastAPI.
#
# Acesse em: http://localhost:7860
# ============================================================

import os
import requests
import gradio as gr

# URL da API (vem da variável de ambiente; padrão para desenvolvimento local)
API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_status() -> dict | None:
    """Verifica se a API está no ar e retorna o status dos serviços."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_example_questions() -> list[str]:
    """Busca exemplos de perguntas da API."""
    try:
        resp = requests.get(f"{API_URL}/chat/examples", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("examples", [])
    except Exception:
        pass
    return [
        "Quais são as causas mais comuns de superaquecimento do motor?",
        "O que indica pressão baixa do óleo?",
        "Meu carro tem 80.000km, o que verificar preventivamente?",
    ]


def chat_with_api(message: str, history: list, show_sources: bool) -> tuple:
    """
    Envia a pergunta para a API e retorna a resposta.

    Args:
        message: Texto digitado pelo usuário
        history: Histórico da conversa (list of [user, assistant] pairs)
        show_sources: Se True, mostra os documentos usados como contexto

    Returns:
        (resposta_formatada, historico_atualizado, info_fontes)
    """
    if not message.strip():
        return history, ""

    # Adiciona a mensagem do usuário ao histórico
    history = history + [[message, None]]

    try:
        # Chama a API usando streaming para mostrar a resposta em tempo real
        response_text = ""
        sources_info = ""

        with requests.post(
            f"{API_URL}/chat",
            json={"question": message, "min_score": 0.25},
            timeout=120,
            stream=False,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                response_text = data["answer"]

                # Formata as fontes usadas
                if show_sources and data.get("sources"):
                    sources_lines = [
                        f"**Documentos usados como contexto** ({data['total_docs_retrieved']} total | Modelo: `{data['model']}`)\n"
                    ]
                    for i, src in enumerate(data["sources"], 1):
                        score_pct = int(src["score"] * 100)
                        sources_lines.append(
                            f"**{i}. {src['source_label']}** (relevância: {score_pct}%)\n"
                            f"> {src['text'][:200]}...\n"
                        )
                    sources_info = "\n".join(sources_lines)
            else:
                response_text = f"❌ Erro na API: {resp.status_code} - {resp.text}"

    except requests.exceptions.ConnectionError:
        response_text = (
            "❌ **Não foi possível conectar à API.**\n\n"
            "Verifique se todos os serviços estão rodando:\n"
            "```\ndocker compose ps\n```"
        )
    except requests.exceptions.Timeout:
        response_text = (
            "⏳ **A resposta demorou muito.**\n\n"
            "Isso pode acontecer na primeira pergunta (o modelo LLM está sendo carregado).\n"
            "Aguarde um momento e tente novamente."
        )
    except Exception as e:
        response_text = f"❌ Erro inesperado: {str(e)}"

    # Atualiza o histórico com a resposta
    history[-1][1] = response_text
    return history, sources_info


def build_interface() -> gr.Blocks:
    """Constrói a interface Gradio."""

    # Verifica status ao iniciar
    status = check_api_status()
    examples = get_example_questions()

    if status:
        status_icon = "🟢" if status["status"] == "healthy" else "🟡"
        docs_count = sum(status.get("indexed_documents", {}).values())
        status_text = (
            f"{status_icon} API conectada | "
            f"Milvus: {'✓' if status['services'].get('milvus') else '✗'} | "
            f"PostgreSQL: {'✓' if status['services'].get('postgres') else '✗'} | "
            f"Ollama: {'✓' if status['services'].get('ollama') else '✗'} | "
            f"Documentos indexados: {docs_count:,}"
        )
    else:
        status_text = "🔴 API não está respondendo - verifique com: docker compose ps"

    with gr.Blocks(
        title="AutoPredict AI",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
            .gradio-container { max-width: 900px !important; }
            .status-bar { font-size: 0.85em; color: #666; padding: 8px; border-radius: 4px; background: #f5f5f5; }
        """,
    ) as demo:

        # ── Header ──────────────────────────────────────────
        gr.Markdown(
            """
            # 🚗 AutoPredict AI
            **Diagnóstico Preditivo Automotivo com IA**

            Faça perguntas sobre manutenção, diagnóstico de falhas e cuidados preventivos.
            O sistema consulta 3 bases de dados especializadas para dar respostas fundamentadas.
            """
        )

        with gr.Row():
            gr.HTML(f'<div class="status-bar">{status_text}</div>')

        # ── Chat ────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversa",
                    height=450,
                    show_copy_button=True,
                    bubble_full_width=False,
                )

                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="Ex: Meu motor está superaquecendo, o que pode ser?",
                        label="Sua pergunta",
                        lines=2,
                        scale=4,
                    )
                    with gr.Column(scale=1):
                        send_btn = gr.Button("Perguntar 🔍", variant="primary")
                        clear_btn = gr.Button("Limpar 🗑️")

                show_sources = gr.Checkbox(
                    label="📚 Mostrar documentos usados como contexto",
                    value=False,
                )

            with gr.Column(scale=1):
                sources_display = gr.Markdown(
                    label="Fontes",
                    value="*As fontes aparecerão aqui quando você fizer uma pergunta.*",
                )

        # ── Exemplos ─────────────────────────────────────────
        gr.Markdown("### 💡 Exemplos de perguntas")
        gr.Examples(
            examples=[[ex] for ex in examples],
            inputs=[question_input],
            label="Clique para usar:",
        )

        # ── Como funciona ────────────────────────────────────
        with gr.Accordion("ℹ️ Como funciona o AutoPredict AI?", open=False):
            gr.Markdown(
                """
                ### Arquitetura RAG Multi-Dataset

                O AutoPredict AI usa **RAG (Retrieval-Augmented Generation)** com 3 bases de dados:

                | Fonte | Conteúdo |
                |-------|----------|
                | 📋 Histórico de Manutenção | Registros de serviços, peças trocadas, histórico por veículo |
                | 📊 Sensores Preditivos | Leituras de temperatura, pressão, vibração e detecção de falhas |
                | ⚠️ Diagnósticos de Motor | Códigos de falha (DTC), severidade e ações recomendadas |

                **Fluxo de uma pergunta:**
                1. Sua pergunta é convertida em vetor numérico (embedding)
                2. O sistema busca os documentos mais similares nas 3 bases
                3. Os documentos relevantes + sua pergunta são enviados ao LLM
                4. O LLM gera uma resposta fundamentada nos dados reais

                **Por que é melhor que perguntar direto ao ChatGPT?**
                Porque as respostas são baseadas nos **seus dados reais** de frota,
                não em conhecimento genérico. Não há alucinação sobre dados que não existem.
                """
            )

        # ── Eventos ──────────────────────────────────────────
        def submit(message, history, show_src):
            new_history, sources = chat_with_api(message, history, show_src)
            return new_history, sources, ""  # "" limpa o input

        send_btn.click(
            fn=submit,
            inputs=[question_input, chatbot, show_sources],
            outputs=[chatbot, sources_display, question_input],
        )

        question_input.submit(
            fn=submit,
            inputs=[question_input, chatbot, show_sources],
            outputs=[chatbot, sources_display, question_input],
        )

        clear_btn.click(
            fn=lambda: ([], "*As fontes aparecerão aqui quando você fizer uma pergunta.*"),
            outputs=[chatbot, sources_display],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",  # Necessário para funcionar no Docker
        server_port=7860,
        show_error=True,
    )
