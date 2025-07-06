# agents/modal_orchestrator.py
import datetime
import uuid
import logging
import modal
from agents.modal_agents import process_editor_agent, process_critic_agent, process_expert_agent
from modal_app import encode_user_query, search_similar_movies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Переменная, означающая завершённый поиск и ожидание нового запроса
WAITING_NEW_QUERY = "waiting_new_query"


class ModalMovieSearchOrchestrator:
    def __init__(self):
        # Все агенты теперь на Modal, не нужна локальная инициализация
        # Подключение к deployed Modal функциям
        try:
            self.encode_func = modal.Function.from_name("tmdb-project", "encode_user_query")
            self.search_func = modal.Function.from_name("tmdb-project", "search_similar_movies")
            logger.info("✅ Successfully connected to deployed Modal functions")
            self.functions_available = True
        except Exception as e:
            logger.error(f"❌ Deployed functions not found: {e}")
            raise RuntimeError(f"Modal functions not available: {e}")

        self.conversation_state = {
            "step": "initial",
            "session_id": self._generate_session_id(),
            "original_plot": "",
            "improved_plot": "",
            "movie_overview": "",
            "search_results": [],
            "final_recommendations": [],
            "processing_metrics": {},
            "started_at": datetime.datetime.utcnow().isoformat() + "Z"
        }

    async def process_user_input(self, user_input: str, user_feedback: str = "") -> dict:
        """Главная функция обработки с Modal агентами"""

        step = self.conversation_state["step"]

        # если предыдущий цикл завершён – делаем «мягкий» reset и запускаем новый
        if step == WAITING_NEW_QUERY and user_input.strip():
            self.reset_conversation()  # «мягкий» сброс счётчиков
            # Продолжаем как с первоначальным вводом
            return await self._handle_initial_input(user_input)

        if step in ("initial", "initial_fallback"):
            return await self._handle_initial_input(user_input)
        # elif self.conversation_state["step"] == "overview_review":
        #     return await self._handle_overview_feedback(user_feedback)
        if step == "refinement_review":
            return await self._handle_initial_input(user_feedback)
            # return await self._handle_refinement_feedback(user_feedback)

        return {"error": f"Unknown conversation state: {step}"}

    async def _handle_initial_input(self, user_input: str) -> dict:
        """Обработка первоначального ввода через Modal EditorAgent"""
        try:
            # # 1. Улучшение текста
            # ВЫЗОВ Modal функции EditorAgent с Nebius API
            logger.info("ВЫЗОВ Modal функции EditorAgent")
            editor_result = process_editor_agent.remote(user_input)
            logger.info(f"Результат работы функции EditorAgent: {editor_result}")

            # ✅ Обработка случаев, когда текст не прошел начальную проверку
            # (недостаточная длина текста)
            if not editor_result.get("approved", False):
                return {
                    "status": "insufficient_length",
                    "message": editor_result["message"],
                    "improved_text": editor_result.get("improved_text", user_input),
                    "step": "initial"
                }

            # Сохранение состояния
            self.conversation_state["original_plot"] = user_input
            improved_text = editor_result.get("improved_text", user_input)
            self.conversation_state["improved_plot"] = improved_text

            # 2. Генерация overview
            # ВЫЗОВ Modal функции FilmCriticAgent с Nebius API
            logger.info("ВЫЗОВ Modal функции FilmCriticAgent")
            overview_result = process_critic_agent.remote(
                plot_description=improved_text,
                action="create"
            )
            logger.info(f"Результат работы функции FilmCriticAgent: {overview_result}")

            movie_overview = overview_result["overview"]
            self.conversation_state["movie_overview"] = movie_overview
            # self.conversation_state["step"] = "overview_review"

            # 3. Поиск фильмов (без ожидания подтверждения)
            encoding_result = self.encode_func.remote(
                self.conversation_state["movie_overview"],
                # self.conversation_state["improved_plot"],
                remove_entities=True
            )
            logger.info(f"Описание пользователя после препроцессинга в FilmCriticAgent: "
                        f"{encoding_result['narrative_features']}")

            search_results = self.search_func.remote(
                encoding_result["embedding"],
                encoding_result["narrative_features"],
                top_k=50,
                rerank_top_n=10
            )

            logger.info(f"Результат поиска фильмов по запросу пользователя в FilmCriticAgent (trimmed): %s",
                        self._trim_search_results(search_results['results'])
                        )
            self.conversation_state["search_results"] = search_results["results"]

            # 4. Экспертный анализ
            expert_result = process_expert_agent.remote(
                user_query=self.conversation_state["movie_overview"],
                search_results=search_results["results"]
            )
            recommendations = expert_result["explanations"]
            self.conversation_state["final_recommendations"] = recommendations
            # Сохраняем результат и переводим оркестратор в состояние ожидания нового поиска
            self.conversation_state["step"] = WAITING_NEW_QUERY
            # self.conversation_state["step"] = "completed"

            return {
                "status": "search_completed",  # ✅ Сразу завершенный поиск
                "next_hint": "Enter a new plot description to start another search",  # будет показано пользователю
                "original_plot": user_input,
                "improved_plot": improved_text,
                "movie_overview": movie_overview,
                "recommendations": recommendations,
                "total_analyzed": len(search_results["results"]),
                "performance_metrics": search_results.get("performance_metrics", {}),
                "step": "completed"
            }

        except Exception as e:
            logger.error(f"Error in initial input handling: {e}")
            # Fallback для любых ошибок редактора
            return {
                "status": "error",
                "message": f"Modal agent processing error: {e}",
                "step": "initial"
            }

    async def _handle_overview_feedback(self, feedback: str) -> dict:
        """Обработка фидбека по overview"""
        if self._is_positive_feedback(feedback):
            return await self._perform_movie_search()
        else:
            # ВЫЗОВ Modal функции для доработки overview
            refined_result = process_critic_agent.remote(
                plot_description=self.conversation_state["movie_overview"],
                action="refine",
                feedback=feedback
            )

            refined_overview = refined_result["overview"]
            self.conversation_state["movie_overview"] = refined_overview
            self.conversation_state["step"] = "refinement_review"

            return {
                "status": "overview_refined",
                "refined_overview": refined_overview,
                "message": "Here's the refined overview. Do you approve it now?",
                "step": "refinement_review"
            }

    async def _handle_refinement_feedback(self, feedback: str) -> dict:
        if self._is_positive_feedback(feedback):
            return await self._perform_movie_search()
        else:
            return await self._handle_overview_feedback(feedback)

    async def _perform_movie_search(self) -> dict:
        """Поиск фильмов через Modal"""
        try:
            # Кодирование запроса (уже на Modal)
            # encoding_result = encode_user_query.remote(
            #     self.conversation_state["movie_overview"],
            #     remove_entities=True
            # )
            encoding_result = self.encode_func.remote(
                self.conversation_state["movie_overview"],
                remove_entities=True
            )

            # Поиск фильмов (уже на Modal)
            search_results = self.search_func.remote(
                encoding_result["embedding"],
                encoding_result["narrative_features"],
                top_k=50,
                rerank_top_n=10
            )

            self.conversation_state["search_results"] = search_results["results"]

            # ВЫЗОВ Modal функции ExpertAgent с Nebius API
            expert_result = process_expert_agent.remote(
                user_query=self.conversation_state["movie_overview"],
                search_results=search_results["results"]
            )

            recommendations = expert_result["explanations"]
            self.conversation_state["final_recommendations"] = recommendations
            self.conversation_state["step"] = "completed"

            return {
                "status": "search_completed",
                "recommendations": recommendations,
                "total_analyzed": len(search_results["results"]),
                "performance_metrics": search_results.get("performance_metrics", {}),
                "step": "completed"
            }

        except Exception as e:
            logger.error(f"Error in movie search: {e}")
            return {
                "status": "error",
                "message": f"Search error: {e}",
                "step": "search_failed"
            }

    @staticmethod
    def _is_positive_feedback(feedback: str) -> bool:
        positive_words = ["approve", "yes", "looks good", "great", "ok", "fine"]
        return any(word in feedback.lower() for word in positive_words)

    @staticmethod
    def _generate_session_id() -> str:
        return str(uuid.uuid4())[:8]

    def reset_conversation(self):
        self.conversation_state = {
            "step": "initial",
            "session_id": self._generate_session_id(),
            "original_plot": "",
            "improved_plot": "",
            "movie_overview": "",
            "search_results": [],
            "final_recommendations": [],
            "processing_metrics": {},
            "started_at": datetime.datetime.utcnow().isoformat() + "Z"
        }

    def get_conversation_summary(self) -> dict:
        """Получение сводки с дополнительным логированием"""
        trimmed = self._trim_search_results(
            self.conversation_state.get("search_results", [])
        )
        logger.info(
            "Getting conversation summary (trimmed search_results): %s",
            trimmed
        )
        return {
            "session_id": self.conversation_state.get("session_id", "N/A"),
            "current_step": self.conversation_state.get("step", "initial"),
            "started_at": self.conversation_state.get("started_at", "N/A"),
            "has_plot": bool(self.conversation_state.get("original_plot", "")),
            "has_overview": bool(self.conversation_state.get("movie_overview", "")),
            "has_recommendations": bool(self.conversation_state.get("final_recommendations", [])),
            "processing_metrics": self.conversation_state.get("processing_metrics", {}),
            "total_search_results": len(self.conversation_state.get("search_results", []))
        }

    @staticmethod
    def _trim_search_results(results: list) -> list:
        """Функция для логирования. Оставляет в movie_data только id, title, narrative_features."""
        wanted = {"id", "title", "narrative_features"}
        trimmed = []

        for item in results:
            # Копируем верхний уровень без movie_data
            base = {k: v for k, v in item.items() if k != "movie_data"}
            # Сужаем movie_data
            md = item.get("movie_data", {})
            base["movie_data"] = {k: md.get(k) for k in wanted if k in md}
            trimmed.append(base)

        return trimmed
