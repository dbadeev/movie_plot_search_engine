# agents/modal_agents.py
import modal
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("movie-agents-nebius")

# Образ с зависимостями для агентов
agents_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "llama-index-core>=0.10.0",   # Базовый пакет (включает core + основные интеграции)
        "llama-index-llms-openai>=0.1.0",       # Для OpenAI-совместимых API
        # "llama-index-llms-llama-api>=0.1.0",
        # "llama-index-agent-react>=0.2.0",
        "openai>=1.0.0",
        "requests>=2.31.0"
    )
)


@app.function(
    image=agents_image,
    secrets=[modal.Secret.from_name("nebius-secret")],
    timeout=300
)
def process_editor_agent(user_text: str, use_react: bool = False) -> dict:
    """EditorAgent обработка на Modal с Nebius API"""
    from agents.editor_agent import EditorAgent

    # Получаем API ключ из Modal Secret
    nebius_api_key = os.environ.get("NEBIUS_API_KEY")
    if not nebius_api_key:
        raise ValueError("NEBIUS_API_KEY not found in Modal secrets")

    # Инициализируем агента с Nebius API
    # ✅ ПРЯМЫЕ ВЫЗОВЫ (без ReAct) по умолчанию
    editor = EditorAgent(nebius_api_key, use_react=use_react)
    # editor = EditorAgent()

    # Обработка текста
    result = editor.process_and_improve_text(user_text)
    logger.info(f"result: {result}")
    return result


@app.function(
    image=agents_image,
    secrets=[modal.Secret.from_name("nebius-secret")],
    timeout=300
)
def process_critic_agent(plot_description: str, action: str = "create", feedback: str = None) -> dict:
    """FilmCriticAgent обработка на Modal с Nebius API"""
    from agents.critic_agent_nebius import FilmCriticAgent

    nebius_api_key = os.environ.get("NEBIUS_API_KEY")
    if not nebius_api_key:
        raise ValueError("NEBIUS_API_KEY not found in Modal secrets")

    critic = FilmCriticAgent(nebius_api_key)

    if action == "create":
        result = critic.create_overview(plot_description)
    elif action == "refine" and feedback:
        result = critic.refine_with_feedback(plot_description, feedback)
    else:
        raise ValueError("Invalid action or missing feedback for refine action")

    return result


@app.function(
    image=agents_image,
    secrets=[modal.Secret.from_name("nebius-secret")],
    timeout=300
)
def process_expert_agent(user_query: str, search_results: list) -> dict:
    """ExpertAgent обработка на Modal с Nebius API"""
    from agents.expert_agent import ExpertAgent

    nebius_api_key = os.environ.get("NEBIUS_API_KEY")
    if not nebius_api_key:
        raise ValueError("NEBIUS_API_KEY not found in Modal secrets")

    expert = ExpertAgent(nebius_api_key)

    result = expert.analyze_and_recommend(user_query, search_results)

    return result
