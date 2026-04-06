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

# ── Paleta de cores ────────────────────────────────────────────────────────────
# Azul marinho: #0D1B2A  /  Dourado: #C9A84C  /  Azul médio: #1B3A5C
# Azul claro:  #2E6DA4  /  Creme:   #F5F0E8  /  Cinza escuro: #1E2D3D
NAVY   = "#0D1B2A"
GOLD   = "#C9A84C"
GOLD_L = "#E2C070"
BLUE_M = "#1B3A5C"
BLUE_L = "#2E6DA4"
CREAM  = "#F5F0E8"
DARK   = "#1E2D3D"
WHITE  = "#FFFFFF"


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
        "Qual a pressão ideal dos pneus?",
        "Como saber se o freio precisa de manutenção?",
        "O que causa vibração excessiva no motor?",
    ]


def _build_status_html(status: dict | None) -> str:
    """Gera o HTML do banner de status dos serviços."""
    if status is None:
        return """
        <div class="ap-status ap-status--error">
            <span class="ap-status__dot"></span>
            <strong>API offline</strong> — verifique os containers com <code>docker compose ps</code>
        </div>"""

    docs_count = sum(status.get("indexed_documents", {}).values())
    svc = status.get("services", {})
    ok = status["status"] == "healthy"

    def badge(label: str, up: bool) -> str:
        cls = "ok" if up else "fail"
        icon = "✓" if up else "✗"
        return f'<span class="ap-badge ap-badge--{cls}">{icon} {label}</span>'

    return f"""
    <div class="ap-status {'ap-status--ok' if ok else 'ap-status--warn'}">
        <span class="ap-status__dot"></span>
        <strong>{'Sistema operacional' if ok else 'Atenção: serviço degradado'}</strong>
        &nbsp;·&nbsp;
        {badge('Milvus', svc.get('milvus', False))}
        {badge('PostgreSQL', svc.get('postgres', False))}
        {badge('Ollama', svc.get('ollama', False))}
        &nbsp;·&nbsp;
        <span class="ap-docs-count">📄 {docs_count:,} documentos indexados</span>
    </div>"""


def _build_metric_cards(status: dict | None) -> str:
    """Gera cards com métricas do sistema."""
    if status is None:
        return ""
    docs = status.get("indexed_documents", {})
    maintenance = docs.get("vehicle_maintenance", 0)
    predictive  = docs.get("car_predictive", 0)
    engine      = docs.get("engine_fault", 0)
    total       = maintenance + predictive + engine
    return f"""
    <div class="ap-metrics">
        <div class="ap-metric-card">
            <div class="ap-metric-icon">📋</div>
            <div class="ap-metric-value">{maintenance:,}</div>
            <div class="ap-metric-label">Registros de Manutenção</div>
        </div>
        <div class="ap-metric-card">
            <div class="ap-metric-icon">📊</div>
            <div class="ap-metric-value">{predictive:,}</div>
            <div class="ap-metric-label">Dados de Sensores</div>
        </div>
        <div class="ap-metric-card">
            <div class="ap-metric-icon">⚠️</div>
            <div class="ap-metric-value">{engine:,}</div>
            <div class="ap-metric-label">Diagnósticos de Motor</div>
        </div>
        <div class="ap-metric-card ap-metric-card--highlight">
            <div class="ap-metric-icon">🗂️</div>
            <div class="ap-metric-value">{total:,}</div>
            <div class="ap-metric-label">Total Indexado</div>
        </div>
    </div>"""


def chat_with_api(message: str, history: list, show_sources: bool):
    """
    Envia a pergunta para a API com streaming real-time.
    É um GENERATOR: faz yield de estados parciais enquanto o LLM responde.

    Args:
        message: Texto digitado pelo usuário
        history: Histórico da conversa (list of [user, assistant] pairs)
        show_sources: Se True, busca e exibe os documentos fonte ao final

    Yields:
        (history_atualizado, sources_info, input_limpo)
    """
    if not message.strip():
        gr.Warning("✏️ Por favor, digite uma pergunta antes de enviar.")
        yield history, "", message
        return

    # Adiciona a mensagem do usuário com indicador "digitando..."
    current_history = history + [[message, "⏳ _Consultando base de dados e gerando resposta..._"]]
    yield current_history, "", ""

    response_text = ""
    sources_info = ""

    try:
        # ── Streaming via GET /chat/stream ──────────────────────────────
        with requests.get(
            f"{API_URL}/chat/stream",
            params={"question": message, "min_score": 0.20},
            stream=True,
            timeout=200,
        ) as resp:
            if resp.status_code == 200:
                for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        response_text += chunk
                        current_history[-1][1] = response_text
                        yield current_history, sources_info, ""
            else:
                response_text = f"❌ Erro na API: {resp.status_code} - {resp.text}"
                current_history[-1][1] = response_text
                yield current_history, sources_info, ""
                return

        # ── Busca fontes via POST /chat (após streaming terminar) ────────
        if show_sources:
            try:
                src_resp = requests.post(
                    f"{API_URL}/chat",
                    json={"question": message, "min_score": 0.20},
                    timeout=200,
                )
                if src_resp.status_code == 200:
                    data = src_resp.json()
                    if data.get("sources"):
                        lines = [
                            f"**Contexto usado** ({data['total_docs_retrieved']} docs | Modelo: `{data['model']}`)\n"
                        ]
                        for i, src in enumerate(data["sources"], 1):
                            score_pct = int(src["score"] * 100)
                            lines.append(
                                f"**{i}. {src['source_label']}** (relevância: {score_pct}%)\n"
                                f"> {src['text'][:200]}...\n"
                            )
                        sources_info = "\n".join(lines)
            except Exception:
                pass  # fontes são opcionais

    except requests.exceptions.ConnectionError:
        response_text = (
            "❌ **Não foi possível conectar à API.**\n\n"
            "Verifique se todos os serviços estão rodando:\n"
            "```\ndocker compose ps\n```"
        )
        current_history[-1][1] = response_text
    except requests.exceptions.Timeout:
        response_text = (
            "⏳ **A resposta demorou muito.**\n\n"
            "Isso pode acontecer na primeira pergunta (modelo LLM sendo carregado).\n"
            "Aguarde um momento e tente novamente."
        )
        current_history[-1][1] = response_text
    except Exception as e:
        current_history[-1][1] = f"❌ Erro inesperado: {str(e)}"

    yield current_history, sources_info, ""


def build_interface() -> gr.Blocks:
    """Constrói a interface Gradio."""

    status  = check_api_status()
    examples = get_example_questions()

    # ── CSS customizado ──────────────────────────────────────────────────
    CSS = f"""
    /* ── Reset / base ─────────────────────────────── */
    :root {{
        --navy:   {NAVY};
        --gold:   {GOLD};
        --gold-l: {GOLD_L};
        --blue-m: {BLUE_M};
        --blue-l: {BLUE_L};
        --cream:  {CREAM};
        --dark:   {DARK};
        --radius: 12px;
        --shadow: 0 4px 24px rgba(0,0,0,0.18);
    }}

    /* ── Layout root ───────────────────────────────── */
    .gradio-container {{
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding: 0 16px !important;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
        background: var(--navy) !important;
    }}
    body, .dark {{
        background: var(--navy) !important;
    }}

    /* ── Header hero ────────────────────────────────── */
    .ap-hero {{
        background: linear-gradient(135deg, var(--navy) 0%, var(--blue-m) 60%, #14324f 100%);
        border-bottom: 3px solid var(--gold);
        padding: 36px 40px 28px;
        border-radius: var(--radius) var(--radius) 0 0;
        position: relative;
        overflow: hidden;
    }}
    .ap-hero::before {{
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(201,168,76,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }}
    .ap-hero-logo {{
        font-size: 2.6rem;
        font-weight: 800;
        color: {WHITE};
        letter-spacing: -0.5px;
        line-height: 1.1;
        margin: 0;
    }}
    .ap-hero-logo span {{
        color: var(--gold);
    }}
    .ap-hero-tagline {{
        color: #a8c4dc;
        font-size: 1.05rem;
        margin-top: 6px;
        font-weight: 400;
    }}
    .ap-hero-pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
    }}
    .ap-pill {{
        background: rgba(201,168,76,0.15);
        border: 1px solid rgba(201,168,76,0.35);
        color: var(--gold-l);
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
        letter-spacing: 0.3px;
    }}

    /* ── Status bar ─────────────────────────────────── */
    .ap-status {{
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 0.84rem;
        margin: 14px 0 0;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        color: #cfd8e3;
    }}
    .ap-status--ok   {{ border-color: rgba(46,189,89,0.4);  }}
    .ap-status--warn {{ border-color: rgba(234,179,8,0.4);  }}
    .ap-status--error{{ border-color: rgba(239,68,68,0.4);  color: #fca5a5; }}
    .ap-status__dot {{
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 6px #22c55e;
        flex-shrink: 0;
    }}
    .ap-status--warn .ap-status__dot  {{ background: #eab308; box-shadow: 0 0 6px #eab308; }}
    .ap-status--error .ap-status__dot {{ background: #ef4444; box-shadow: 0 0 6px #ef4444; }}
    .ap-badge {{
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    .ap-badge--ok   {{ background: rgba(34,197,94,0.15);  color: #86efac; }}
    .ap-badge--fail {{ background: rgba(239,68,68,0.15);  color: #fca5a5; }}
    .ap-docs-count  {{ color: var(--gold-l); font-weight: 600; }}

    /* ── Metric cards ───────────────────────────────── */
    .ap-metrics {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
        margin: 18px 0 0;
    }}
    .ap-metric-card {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: var(--radius);
        padding: 16px 14px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }}
    .ap-metric-card:hover {{
        transform: translateY(-2px);
        border-color: rgba(201,168,76,0.4);
    }}
    .ap-metric-card--highlight {{
        border-color: rgba(201,168,76,0.35);
        background: rgba(201,168,76,0.06);
    }}
    .ap-metric-icon  {{ font-size: 1.5rem; margin-bottom: 6px; }}
    .ap-metric-value {{ font-size: 1.6rem; font-weight: 700; color: var(--gold); }}
    .ap-metric-label {{ font-size: 0.75rem; color: #8ca0b8; margin-top: 2px; }}

    /* ── Chat container ─────────────────────────────── */
    .ap-chat-wrap {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: var(--radius);
        padding: 4px;
        margin-top: 18px;
    }}

    /* ── Chatbot messages ───────────────────────────── */
    .message.user .bubble-wrap .md {{
        background: linear-gradient(135deg, var(--blue-m), var(--blue-l)) !important;
        color: {WHITE} !important;
        border-radius: 16px 16px 4px 16px !important;
        padding: 12px 16px !important;
        box-shadow: var(--shadow) !important;
    }}
    .message.bot .bubble-wrap .md {{
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(201,168,76,0.2) !important;
        color: #dde6f0 !important;
        border-radius: 16px 16px 16px 4px !important;
        padding: 12px 16px !important;
    }}
    /* Avatar user */
    .message.user .icon-wrap svg {{ fill: var(--gold) !important; }}

    /* ── Input box ──────────────────────────────────── */
    .ap-input-row textarea {{
        background: rgba(255,255,255,0.07) !important;
        border: 1.5px solid rgba(201,168,76,0.3) !important;
        border-radius: 10px !important;
        color: {WHITE} !important;
        font-size: 0.97rem !important;
        resize: none !important;
        transition: border-color 0.2s !important;
    }}
    .ap-input-row textarea:focus {{
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 3px rgba(201,168,76,0.15) !important;
    }}
    .ap-input-row textarea::placeholder {{
        color: #5d7a96 !important;
    }}

    /* ── Buttons ────────────────────────────────────── */
    .ap-btn-send {{
        background: linear-gradient(135deg, var(--gold), #a8832a) !important;
        color: var(--navy) !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0 24px !important;
        height: 48px !important;
        cursor: pointer !important;
        transition: opacity 0.2s, transform 0.15s !important;
        box-shadow: 0 2px 12px rgba(201,168,76,0.3) !important;
        white-space: nowrap !important;
    }}
    .ap-btn-send:hover  {{ opacity: 0.88 !important; transform: translateY(-1px) !important; }}
    .ap-btn-send:active {{ transform: scale(0.97) !important; }}

    .ap-btn-clear {{
        background: rgba(255,255,255,0.06) !important;
        color: #8ca0b8 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        height: 48px !important;
        font-size: 0.88rem !important;
        transition: background 0.2s !important;
        white-space: nowrap !important;
    }}
    .ap-btn-clear:hover {{ background: rgba(239,68,68,0.12) !important; color: #fca5a5 !important; }}

    /* ── Checkbox ───────────────────────────────────── */
    .ap-checkbox label {{
        color: #8ca0b8 !important;
        font-size: 0.87rem !important;
    }}

    /* ── Sources panel ──────────────────────────────── */
    .ap-sources {{
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(201,168,76,0.2) !important;
        border-radius: var(--radius) !important;
        padding: 16px !important;
        height: 100% !important;
        min-height: 200px;
    }}
    .ap-sources p, .ap-sources li, .ap-sources blockquote {{
        color: #8ca0b8 !important;
        font-size: 0.84rem !important;
        line-height: 1.5 !important;
    }}
    .ap-sources strong {{ color: var(--gold-l) !important; }}
    .ap-sources blockquote {{
        border-left: 3px solid var(--gold) !important;
        padding-left: 10px !important;
        margin: 4px 0 !important;
        background: rgba(255,255,255,0.02) !important;
        border-radius: 0 6px 6px 0 !important;
    }}
    .ap-sources code {{
        background: rgba(201,168,76,0.12) !important;
        color: var(--gold-l) !important;
        padding: 1px 5px !important;
        border-radius: 4px !important;
    }}

    /* ── Section headings ───────────────────────────── */
    .ap-section-title {{
        color: var(--gold);
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin: 22px 0 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .ap-section-title::after {{
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(201,168,76,0.3), transparent);
    }}

    /* ── Example chips ──────────────────────────────── */
    .ap-examples-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
        gap: 10px;
        margin-top: 4px;
    }}
    .ap-example-chip {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(201,168,76,0.2);
        border-radius: 8px;
        padding: 10px 14px;
        cursor: pointer;
        color: #b0c4d8;
        font-size: 0.85rem;
        line-height: 1.4;
        transition: background 0.18s, border-color 0.18s, color 0.18s;
        text-align: left;
        width: 100%;
    }}
    .ap-example-chip:hover {{
        background: rgba(201,168,76,0.1);
        border-color: var(--gold);
        color: {WHITE};
    }}

    /* ── Accordion ──────────────────────────────────── */
    .ap-accordion {{
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: var(--radius) !important;
        margin-top: 18px !important;
    }}
    .ap-accordion .label-wrap span {{
        color: #8ca0b8 !important;
        font-size: 0.9rem !important;
    }}
    .ap-accordion > .wrap > .panel {{
        background: transparent !important;
    }}
    .ap-accordion .md p, .ap-accordion .md li {{
        color: #8ca0b8 !important;
        font-size: 0.87rem !important;
        line-height: 1.6 !important;
    }}
    .ap-accordion .md strong {{ color: var(--gold-l) !important; }}
    .ap-accordion .md table {{
        border-collapse: collapse !important;
        width: 100% !important;
    }}
    .ap-accordion .md th {{
        background: rgba(201,168,76,0.12) !important;
        color: var(--gold-l) !important;
        padding: 8px 12px !important;
        font-size: 0.82rem !important;
    }}
    .ap-accordion .md td {{
        padding: 8px 12px !important;
        border-bottom: 1px solid rgba(255,255,255,0.05) !important;
        color: #8ca0b8 !important;
        font-size: 0.82rem !important;
    }}
    .ap-accordion .md ol li::marker {{
        color: var(--gold) !important;
    }}

    /* ── Footer ─────────────────────────────────────── */
    .ap-footer {{
        text-align: center;
        color: #3d5570;
        font-size: 0.78rem;
        padding: 18px 0 10px;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 28px;
    }}
    .ap-footer a {{ color: var(--gold); text-decoration: none; }}

    /* ── Responsive ─────────────────────────────────── */
    @media (max-width: 768px) {{
        .ap-hero {{ padding: 24px 20px 20px; }}
        .ap-hero-logo {{ font-size: 1.9rem; }}
        .ap-metrics {{ grid-template-columns: repeat(2, 1fr); }}
        .ap-examples-grid {{ grid-template-columns: 1fr; }}
        .gradio-container {{ padding: 0 8px !important; }}
    }}
    @media (max-width: 480px) {{
        .ap-hero-logo {{ font-size: 1.5rem; }}
        .ap-hero-tagline {{ font-size: 0.9rem; }}
        .ap-metrics {{ grid-template-columns: repeat(2, 1fr); gap: 8px; }}
        .ap-metric-value {{ font-size: 1.3rem; }}
    }}

    /* ── Gradio overrides ───────────────────────────── */
    footer {{ display: none !important; }}
    .gradio-container .prose h1,
    .gradio-container .prose h2,
    .gradio-container .prose h3 {{
        color: var(--gold) !important;
    }}
    .chatbot .wrap {{ background: transparent !important; }}
    .chatbot {{ background: transparent !important; }}
    label.svelte-1b6s6vi {{ color: #8ca0b8 !important; }}
    """

    with gr.Blocks(
        title="AutoPredict AI — Diagnóstico Automotivo",
        css=CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        # ── Hero header ──────────────────────────────────────────────────
        gr.HTML(f"""
        <div class="ap-hero">
            <h1 class="ap-hero-logo">🚗 Auto<span>Predict</span> AI</h1>
            <p class="ap-hero-tagline">Diagnóstico Preditivo Automotivo com RAG &amp; LLM Local</p>
            <div class="ap-hero-pills">
                <span class="ap-pill">🔍 RAG Multi-Dataset</span>
                <span class="ap-pill">🤖 LLM Local (llama3.2)</span>
                <span class="ap-pill">⚡ Streaming em Tempo Real</span>
                <span class="ap-pill">📊 3 Bases de Dados</span>
                <span class="ap-pill">🔒 100% Privado</span>
            </div>
        </div>
        """)

        # ── Status + métricas ────────────────────────────────────────────
        status_html   = gr.HTML(_build_status_html(status))
        metrics_html  = gr.HTML(_build_metric_cards(status))

        # ── Área principal de chat ───────────────────────────────────────
        gr.HTML('<div class="ap-section-title">💬 Chat de Diagnóstico</div>')

        with gr.Row(equal_height=True):
            # Coluna do chat (maior)
            with gr.Column(scale=3, min_width=300):
                with gr.Group(elem_classes="ap-chat-wrap"):
                    chatbot = gr.Chatbot(
                        label="",
                        height=440,
                        show_copy_button=True,
                        bubble_full_width=False,
                        avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=autopredict&backgroundColor=1B3A5C"),
                        type="tuples",
                    )

                with gr.Row(equal_height=True, elem_classes="ap-input-row"):
                    question_input = gr.Textbox(
                        placeholder="Ex: Meu motor está superaquecendo, o que pode ser?",
                        label="",
                        lines=2,
                        max_lines=4,
                        scale=5,
                        show_label=False,
                        container=False,
                    )
                    with gr.Column(scale=1, min_width=110):
                        send_btn = gr.Button(
                            "Enviar ➤",
                            variant="primary",
                            elem_classes="ap-btn-send",
                            size="lg",
                        )
                        clear_btn = gr.Button(
                            "Limpar",
                            elem_classes="ap-btn-clear",
                            size="sm",
                        )

                show_sources = gr.Checkbox(
                    label="📚 Mostrar documentos utilizados como contexto",
                    value=False,
                    elem_classes="ap-checkbox",
                )

            # Coluna de fontes (menor)
            with gr.Column(scale=1, min_width=220):
                gr.HTML('<div class="ap-section-title" style="margin-top:0">📁 Fontes</div>')
                sources_display = gr.Markdown(
                    value="*Ative **Mostrar documentos** e faça uma pergunta para ver as fontes utilizadas.*",
                    elem_classes="ap-sources",
                )

        # ── Exemplos de perguntas ────────────────────────────────────────
        gr.HTML('<div class="ap-section-title">💡 Exemplos de Perguntas</div>')

        example_btns = []
        rows_of_4 = [examples[i:i+4] for i in range(0, len(examples), 4)]
        for row_examples in rows_of_4:
            with gr.Row():
                for ex in row_examples:
                    with gr.Column(scale=1, min_width=150):
                        btn = gr.Button(
                            f"💬 {ex}",
                            elem_classes="ap-example-chip",
                            size="sm",
                        )
                        example_btns.append((btn, ex))

        # ── Como funciona ────────────────────────────────────────────────
        with gr.Accordion(
            "ℹ️  Como funciona o AutoPredict AI?",
            open=False,
            elem_classes="ap-accordion",
        ):
            gr.Markdown("""
            ### Arquitetura RAG Multi-Dataset

            O AutoPredict AI usa **RAG (Retrieval-Augmented Generation)** combinando busca semântica vetorial com geração de linguagem natural:

            | Base de Dados | Conteúdo | Documentos |
            |---|---|---|
            | 📋 Histórico de Manutenção | Registros de serviços, peças trocadas, histórico por quilometragem | 5.000 |
            | 📊 Sensores Preditivos | Leituras de temperatura, pressão dos pneus, espessura dos freios | 1.100 |
            | ⚠️ Diagnósticos de Motor | Vibração, temperatura, acústica, pressão de admissão/escape | 5.000 |

            **Fluxo de uma pergunta:**
            1. **Embedding** — sua pergunta é convertida em vetor numérico (384 dimensões)
            2. **Busca vetorial** — o Milvus encontra os documentos mais similares nas 3 bases
            3. **MMR Reranking** — algoritmo de diversidade seleciona os melhores documentos sem repetição
            4. **Geração** — o LLM local (llama3.2:3b) recebe documentos + pergunta e gera a resposta
            5. **Streaming** — os tokens chegam em tempo real, sem esperar a resposta completa

            **Por que usar em vez do ChatGPT?**
            As respostas são baseadas nos **dados reais** dos seus veículos, não em conhecimento genérico.
            O modelo roda **100% local** — nenhum dado sai da sua infraestrutura.
            """)

        # ── Footer ───────────────────────────────────────────────────────
        gr.HTML("""
        <div class="ap-footer">
            AutoPredict AI &mdash; Projeto Acadêmico &middot;
            RAG + LLM local &middot;
            <a href="http://localhost:8000/docs" target="_blank">API Docs</a> &middot;
            <a href="http://localhost:5001" target="_blank">MLflow</a> &middot;
            <a href="http://localhost:9001" target="_blank">MinIO</a>
        </div>
        """)

        # ── Eventos ──────────────────────────────────────────────────────
        send_btn.click(
            fn=chat_with_api,
            inputs=[question_input, chatbot, show_sources],
            outputs=[chatbot, sources_display, question_input],
        )

        question_input.submit(
            fn=chat_with_api,
            inputs=[question_input, chatbot, show_sources],
            outputs=[chatbot, sources_display, question_input],
        )

        clear_btn.click(
            fn=lambda: ([], "*Ative **Mostrar documentos** e faça uma pergunta para ver as fontes utilizadas.*"),
            outputs=[chatbot, sources_display],
        )

        # Chips de exemplo: closure explícita sem parâmetros para o Gradio
        # não inferir inputs errados
        def make_setter(val):
            def _set():
                return val
            return _set

        for btn, text in example_btns:
            btn.click(fn=make_setter(text), inputs=[], outputs=[question_input])

        # Atualiza status a cada carregamento de página
        def _refresh_status():
            s = check_api_status()
            return _build_status_html(s), _build_metric_cards(s)

        demo.load(fn=_refresh_status, outputs=[status_html, metrics_html])

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",  # Necessário para funcionar no Docker
        server_port=7860,
        show_error=True,
    )
