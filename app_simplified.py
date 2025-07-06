# app_simplified.py
import modal
import gradio as gr
import asyncio
import os
import logging

# from modal_app import app
# from agents.orchestrator import SimplifiedMovieSearchOrchestrator

# Импорт Modal оркестратора вместо локального
from agents.modal_orchestrator import ModalMovieSearchOrchestrator

app = modal.App("movie-plot-search")

# print("Trying to lookup functions...")
# try:
#     encode_func = modal.Function.from_name("tmdb-project", "encode_user_query")
#     print("✅ encode_user_query function found")
# except Exception as e:
#     print(f"❌ Error looking up encode_user_query: {e}")


# --- Основная функция запуска приложения ---
def _run_main_app():
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Инициализация Modal оркестратора (без API ключа)
    orchestrator = ModalMovieSearchOrchestrator()

    async def chat_interface(message: str, history: list) -> tuple:
        """ Основной интерфейс чата с Modal агентами + Nebius LLM5(только английский)"""
        try:
            logger.info(f"Processing user message: {message[:50]}...")

            # ВСЕ LLM ВЫЗОВЫ ПРОИСХОДЯТ НА MODAL С NEBIUS API
            result = await orchestrator.process_user_input(message)
            logger.info(f"RESULT: {result}")
            # Формирование ответа
            response_parts = []

            # ---------- 1. Обработка статусов от оркестратора ----------
            status = result.get("status")

            # ✅ Обработка случая недостаточной длины
            # if result.get("status") == "insufficient_length":
            #     response_parts.append("**❌ Text Too Short**")
            #     response_parts.append(result.get("message", ""))

            if status == "insufficient_length":
                response_parts += [
                    "**❗ **Editor Feedback:**",
                    result.get("message", ""),
                    "\n---\nPlot description is too short (min 50 words). "
                    "Please expand your plot description and try again."
                ]

            # (2) Полный успех: найдено 3 фильма + экспертный отчёт
            elif status == "search_completed":
                logger.info(f"**✅ Поиск завершен! Найдены рекомендации фильмов**")
                response_parts.append("**✅ Plot processed and search completed!**")
                # ✅ Показываем improved plot для информации
                if (result.get("improved_plot") and
                        result.get("improved_plot") != result.get("original_plot")):
                    logger.info(f"**📝 Улучшенное описание:** {result.get('improved_plot')}")
                    response_parts.append(f"**📝 Improved plot:** {result.get('improved_plot')}")
                # ✅ Показываем movie overview для информации
                if result.get("movie_overview"):
                    response_parts.append(f"\n**🎬 Generated movie overview:**\n"
                                          f"{result.get('movie_overview')}")
                # ✅ Основной блок рекомендаций с новым форматом
                response_parts.append("\n" + "=" * 60)
                response_parts.append("**🎯 EXPERT SYSTEM RECOMMENDATIONS**")
                response_parts.append("=" * 30)

                # response_parts.append(result.get("recommendations", ""))
                recommendations = result.get("recommendations", "")
                if recommendations:
                    response_parts.append(recommendations)
                else:
                    response_parts.append("No recommendations were generated.")

                # ✅ Метрики производительности
                response_parts.append("\n" + "=" * 60)
                response_parts.append("**📊 PERFORMANCE METRICS**")
                response_parts.append("=" * 60)

                metrics = result.get("performance_metrics", {})
                if metrics:
                    response_parts.append(f"🚀 **GPU Used:** {'✅ Yes' if metrics.get('using_gpu', False) else '❌ No'}")
                    response_parts.append(f"⚡ **Search Time:** {metrics.get('search_time', 0):.3f}s")
                    response_parts.append(f"🔄 **Total Processing Time:** {metrics.get('total_time', 0):.3f}s")
                    response_parts.append(f"🎬 **Movies Analyzed:** {result.get('total_analyzed', 0)}")

                    if result.get('methodology'):
                        response_parts.append(f"🧮 **Methodology:** {result.get('methodology')}")
                    if result.get('evaluation_formula'):
                        response_parts.append(f"📐 **Evaluation Formula:** {result.get('evaluation_formula')}")

                    # Russian comment: приглашение к новому поиску
                    response_parts.append("\n" + "=" * 30)
                    response_parts.append("**🔄 Ready for the next search!**")
                    response_parts.append("Type a new movie plot and I will find more recommendations.")

            # ✅ Обработка ошибок
            elif status == "error":
                response_parts.append("**❌ System Error occurred:**")
                response_parts.append(result.get("message", "Unknown error"))
            else:
                response_parts.append(f"⚠️ Unhandled status: {status}")

            # ---------- 2. Формируем ответ и историю ----------
            assistant_reply = "\n".join(response_parts)
            new_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": assistant_reply}
            ]

            # Автообновление session info после обработки
            if status in ["search_completed", "needs_improvement", "insufficient_length"]:
                # Обновляем session info автоматически
                _ = get_session_info()  # обновит компонент через .then() в Gradio
                logger.info(f"Session info updated: {_}")

                # Очищаем поле ввода
                return new_history, ""

        except Exception as e:
            logger.error(f"Error in chat interface: {e}")
            error_response = f"**❌ System Error:** {str(e)}"

            # Формат messages для ошибок
            # Обработка ошибок
            new_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": f"**❌ Unexpected error:** {e}"}
            ]

            return new_history, ""

    def reset_chat():
        """Сброс чата с логированием"""
        logger.info("Resetting chat session")
        orchestrator.reset_conversation()
        return [], ""       # Возвращаем пустую историю

    def get_session_info():
        """Получение информации о текущей сессии"""
        try:
            # logger.warning(f"Summary type: {type(orchestrator.get_conversation_summary())} _
            # Summary: {orchestrator.get_conversation_summary()}")
            summary = orchestrator.get_conversation_summary()
            logger.info(f"Getting session summary: {summary}")

            return f"""**Hybrid Session Info:**
    - ID: {summary['session_id']}
    - Step: {summary['current_step']}
    - Has Plot: {'✅' if summary.get('has_plot', False) else '❌'}
    - Has Overview: {'✅' if summary.get('has_overview', False) else '❌'}
    - Has Recommendations: {'✅' if summary.get('has_recommendations', False) else '❌'}
    - Total Results: {summary.get('total_search_results', 0)}
            """

        except Exception as e:
            logger.error(f"Error in get_session_info: {e}")
            return f"Error getting session info: {e}"

    def force_refresh_session_info():
        """Принудительное обновление с логированием состояния"""
        try:
            # ✅ Дополнительное логирование для отладки
            logger.info("Force refreshing session info...")
            logger.info(f"Current orchestrator state: {orchestrator.conversation_state}")

            summary = orchestrator.get_conversation_summary()
            logger.info(f"Retrieved summary: {summary}")

            return get_session_info()
        except Exception as e:
            logger.error(f"Error in force refresh: {e}")
            return f"Refresh error: {e}"

    # Создание интерфейса Gradio
    with gr.Blocks(title="🎬 Movie Plot Search", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
            # 🎬 Movie Plot Search Engine
            
            **🏗️ Architecture:**
                 🖥️ **UI**: Local Gradio interface;
                 ⚡ **Agents**: Running on Modal Cloud;
                 🤖 **LLM**: Nebius AI Studio API (Llama-3.3-70B-Instruct).

            ****The essence of the project:**** 
            *Describe the plot of the story in English, and the System will search the database for three films with a 
            similar script.* \n\n
            The system uses multi-agent architecture with GPU acceleration for optimal performance.
            
            **🤖 Powered by:** Nebius AI Studio | Modal Labs | FAISS | LlamaIndex ReAct Agents
        """)

        with gr.Row():
            with gr.Column(scale=4):
                # Добавляем type='messages'
                chatbot = gr.Chatbot(
                    value=[],
                    height=600,
                    label="🎬 Conversation with AI Agents (Local UI → Modal Agents → Nebius LLM)",
                    show_copy_button=True,
                    type='messages'  # Новый формат сообщений
                )

                msg = gr.Textbox(
                    placeholder="Describe a movie plot (50-100 words in English)...",
                    label="Your message",
                    lines=3,
                    max_lines=5
                )

                with gr.Row():
                    submit_btn = gr.Button("🚀 Submit", variant="primary", scale=2)
                    clear_btn = gr.Button("🔄 Clear Chat", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("""
                 ### 🔍 How to use:
                
                1. **📝 Describe the plot** (50-100 words in English)
                2. **✅ Agent Editor validates** (length) and improves your description (grammar)
                3. **🎬 Agent Film Critic** creates a movie overview based on your story
                4. **🔍 System searches** the database for 10 films that correlate with your description 
                5. **🎯 Agent Film Expert selects** top 3 movies with explanations
                
                ### 📋 Requirements:
                - ✅ English text only
                - ✅ 50-100 words
                - ✅ Clear plot description
                - ✅ Proper grammar (AI will help)
                
                ### ⚡ Features:
                - 🚀 FAISS search
                - 🧠 Multi-agent reasoning
                - 📊 Semantic + narrative similarity
                - 🎯 Expert film analysis
                """)

                session_info = gr.Textbox(
                    label="Session Info",
                    value=get_session_info(),
                    interactive=False,
                    lines=5
                )

                refresh_btn = gr.Button("🔄 Refresh Info", size="sm")

        # Обработчики событий
        submit_btn.click(
            fn=chat_interface,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        ).then(  # ✅ ДОБАВЛЕНО: Автообновление после отправки
            fn=get_session_info,
            outputs=[session_info]
        )

        msg.submit(
            fn=chat_interface,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        ).then(  # ✅ ДОБАВЛЕНО: Автообновление после Enter
            fn=get_session_info,
            outputs=[session_info]
        )

        clear_btn.click(
            fn=reset_chat,
            outputs=[chatbot, msg]
        )

        refresh_btn.click(
            fn=force_refresh_session_info,
            outputs=[session_info]
        )

# Запуск приложения
    logger.info("Starting Movie Plot Search application")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        debug=False,
        show_error=True
    )


# --- Функция запуска через Modal (если нужно полностью на Modal)---
@app.function(secrets=[modal.Secret.from_name("nebius-secret")])
def run_app():
    _run_main_app()


# --- Локальный запуск ---
if __name__ == "__main__":
    _run_main_app()
