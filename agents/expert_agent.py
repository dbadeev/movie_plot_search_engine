# agents/expert_agent.py
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
# from llama_index.llms.llama_api import LlamaAPI
from llama_index.core.tools import FunctionTool

from agents.nebius_simple import create_nebius_llm

import datetime
import json
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _create_top3_selection_tool() -> FunctionTool:
    """Инструмент выбора топ-3 фильмов"""

    def select_top3_movies(evaluated_movies: list, **kwargs) -> dict:
        """Select top 3 movies based on comprehensive scores"""
        try:
            # Сортировка по итоговому скору
            sorted_movies = sorted(
                evaluated_movies,
                key=lambda x: x.get('final_score', 0),
                reverse=True
            )

            top3 = sorted_movies[:3]
            trimmed_for_log = [
                ExpertAgent.trim_movie_data(m.get("movie_data", {})) for m in top3
            ]
            logger.info("TOP-3 (trimmed): %s", trimmed_for_log)
            # logger.info(f"Начальный response: {top3}")

            return {
                "top3_movies": top3,
                "selection_criteria": "Comprehensive weighted scoring",
                "total_evaluated": len(evaluated_movies),
                "score_range": {
                    "highest": top3[0].get('final_score', 0) if top3 else 0,
                    "lowest": top3[-1].get('final_score', 0) if top3 else 0
                },
                "selected_at": datetime.datetime.utcnow().isoformat() + "Z"
            }

        except Exception as e:
            return {
                "error": str(e),
                "top3_movies": evaluated_movies[:3] if evaluated_movies else []
            }

    return FunctionTool.from_defaults(
        fn=select_top3_movies,
        name="select_top3",
        description="Choose top 3 movies from evaluated list"
    )


class ExpertAgent:
    def __init__(self, nebius_api_key: str):
        # ✅ Прямой Nebius LLM
        self.llm = create_nebius_llm(
            api_key=nebius_api_key,
            model="meta-llama/Llama-3.3-70B-Instruct-fast",
            # model="deepseek-ai/DeepSeek-R1-fast",
            temperature=0.2
        )

        self.tools = [
            self._create_comprehensive_evaluation_tool(),
            _create_top3_selection_tool(),  # Внешняя функция
            self._create_justification_tool()
            # self._create_movie_analysis_tool(),
            # self._create_recommendation_generation_tool(),
            # self._create_explanation_tool()
            # self._create_relevance_scoring_tool(),
            # self._create_recommendation_tool()
        ]

        # Карта инструментов по имени
        self.tools_map = {t.metadata.name: t for t in self.tools}

    @staticmethod
    def _fallback_response(cb_mgr, exc):
        """Возвращается, когда превышен лимит итераций."""
        return ("TOOL: select_top3\nOBSERVATION: proceed\nTOOL: create_justification\n"
                "OBSERVATION: proceed\nFINAL: Unable to finish in allotted steps; return "
                "generic justifications.")

    @staticmethod
    def _get_system_prompt() -> str:
        """Статический метод для получения системного промпта"""

        return """You are an Expert Film Analysis Agent specializing in movie recommendation and analysis.

Your responsibilities:
# 1. Analyze the semantic and narrative similarity between user queries and movies
# 2. Select the 3 most relevant movies from search results
# 3. Provide detailed justifications for your choices
# 4. Highlight specific plot, thematic, and structural similarities
# 
# Use the Thought-Action-Observation cycle:
# - Think about the key elements of the user's query (themes, plot structure, genre, tone)
# - Analyze each movie candidate for multiple types of relevance
# - Score and rank movies based on comprehensive factors
# - Generate compelling explanations that demonstrate deep understanding
# 
# Focus on: narrative structure, thematic resonance, character dynamics, 
# emotional tone, genre elements, and plot mechanics.

1. Comprehensively evaluate movies using semantic_score, narrative_similarity, and movie metadata
2. Apply weighted scoring formula considering: genres, title relevance, ratings (vote_average, imdb_rating)
3. IGNORE named entities (names, titles, locations) when comparing plots
4. Select TOP 3 movies based on comprehensive evaluation
5. For each movie, provide 4-5 sentence detailed justifications
6. Highlight specific plot, thematic, and structural similarities

Use the Thought-Action-Observation cycle:
- Think about the key elements of the user's query (themes, plot structure, genre, tone)
- Analyze each movie candidate for multiple types of relevance
- Score and rank movies based on comprehensive factors
- Generate compelling explanations that demonstrate deep understanding

Focus on thematic similarity, narrative structure, genre alignment, and quality metrics 
while ignoring specific names and locations.

IMPORTANT: 
- Each justification **must be unique**; compare it with previously generated ones and re-write if too similar.
- Avoid generic phrases like "strong alignment". Provide concrete plot or structural overlaps.
"""

    # ===============================================================
    # Новый вариант функций Эксперта
    def _create_comprehensive_evaluation_tool(self) -> FunctionTool:
        """Инструмент комплексной оценки фильмов по новой формуле"""
        # ✅ Захватываем self.llm в замыкание
        llm = self.llm

        from typing import Annotated

        def evaluate_movie_comprehensive(user_query: Annotated[str, "Original user plot"],
                                         movie_data: Annotated[dict, "Full JSON of ONE movie (title, overview, …)"]) \
                -> dict:
            """Comprehensive movie evaluation using weighted formula. Returns dict with final_score"""

            #  Безопасная распаковка, если movie_data вложенный {'movie1': {...}}, берём первый dict
            if "title" not in movie_data and isinstance(movie_data, dict):
                movie_data = dict(next(iter(movie_data.values())))

            try:
                # Извлечение данных
                movie_title = movie_data.get('title', 'Unknown')
                overview = movie_data.get('overview', '')
                genres = movie_data.get('genres', '')
                vote_average = float(movie_data.get('vote_average', 0))
                imdb_rating = float(movie_data.get('imdb_rating', 0))
                semantic_score = float(movie_data.get('semantic_score', 0))
                narrative_similarity = float(movie_data.get('narrative_similarity', 0))

                # Оценка соответствия жанров через LLM
                genre_prompt = f"""
                Evaluate genre alignment between user query and movie (0.0-1.0):
                User Query: "{user_query}"
                Movie Genres: "{genres}"

                When you evaluate a movie, call :
                    TOOL: comprehensive_evaluation
                    ARGS: {{
                      "user_query": "<copy user_query>",
                      "movie_data": <JSON of ONE movie from search_results>
                    }}

                IGNORE specific names, locations, characters. Focus on thematic content.
                Return only a number between 0.0 and 1.0.
                """

                # ✅ Используем захваченную переменную llm
                genre_response = llm.complete(genre_prompt)
                try:
                    match = re.search(r'[0-9]*\.?[0-9]+', genre_response.text)
                    if match:
                        genre_alignment = float(match.group())
                        genre_alignment = max(0.0, min(1.0, genre_alignment))
                    else:
                        genre_alignment = 0.5
                except (AttributeError, ValueError, TypeError) as e:
                    logger.warning(f"Error parsing genre alignment: {e}")
                    genre_alignment = 0.5

                # Оценка соответствия названия через LLM
                title_prompt = f"""
                Evaluate title relevance to user query (0.0-1.0):
                User Query: "{user_query}"
                Movie Title: "{movie_title}"

                IGNORE exact name matches. Focus on thematic and conceptual relevance.
                Return only a number between 0.0 and 1.0.
                """

                # ✅ Используем захваченную переменную llm
                title_response = llm.complete(title_prompt)
                try:
                    match = re.search(r'[0-9]*\.?[0-9]+', title_response.text)
                    if match:
                        title_relevance = float(match.group())
                        title_relevance = max(0.0, min(1.0, title_relevance))
                    else:
                        title_relevance = 0.3
                except (AttributeError, ValueError, TypeError) as e:
                    logger.warning(f"Error parsing title relevance: {e}")
                    title_relevance = 0.3

                # Нормализация рейтингов
                normalized_vote_avg = vote_average / 10.0 if vote_average > 0 else 0.5
                normalized_imdb = imdb_rating / 10.0 if imdb_rating > 0 else 0.5

                logger.info(f"Данные для комплексной оценки фильма {movie_title}:  \n"
                            f"{semantic_score} - 65% - семантическое сходство \n"
                            f"{narrative_similarity} - 15% - нарративное сходство \n"
                            f"{genre_alignment} - 4% - жанровое соответствие \n"
                            f"{title_relevance} - 4% - соответствие названия \n"
                            f"{normalized_vote_avg} - 2% - рейтинг TMDB \n"
                            f"{normalized_imdb} - 10% - рейтинг IMDb")
                # Комплексная формула оценки
                final_score = (
                        semantic_score * 0.65 +  # 65% - семантическое сходство
                        narrative_similarity * 0.15 +  # 15% - нарративное сходство
                        genre_alignment * 0.04 +  # 4% - жанровое соответствие
                        title_relevance * 0.04 +  # 4% - соответствие названия
                        normalized_vote_avg * 0.02 +  # 2% - рейтинг TMDB
                        normalized_imdb * 0.10  # 10% - рейтинг IMDb
                )
                logger.info(f"Финальная оценка фильма: {final_score}\n")

                return {
                    "movie_title": movie_title,
                    "final_score": round(final_score, 4),
                    "score_breakdown": {
                        "semantic_score": semantic_score,
                        "narrative_similarity": narrative_similarity,
                        "genre_alignment": genre_alignment,
                        "title_relevance": title_relevance,
                        "normalized_vote_avg": normalized_vote_avg,
                        "normalized_imdb": normalized_imdb
                    },
                    "quality_indicators": {
                        "vote_average": vote_average,
                        "imdb_rating": imdb_rating,
                        "has_high_ratings": vote_average >= 7.0 or imdb_rating >= 7.0
                    },
                    "evaluated_at": datetime.datetime.utcnow().isoformat() + "Z"
                }

            except Exception as e:
                logger.error(f"Error evaluating {movie_data.get('title', 'Unknown')}: {e}")
                return {
                    "movie_title": movie_data.get('title', 'Unknown'),
                    "final_score": 0.3,
                    "error": str(e),
                    "evaluated_at": datetime.datetime.utcnow().isoformat() + "Z"
                }

        return FunctionTool.from_defaults(
            fn=evaluate_movie_comprehensive,
            name="comprehensive_evaluation",
            description="Evaluate one movie using comprehensive weighted formula"
        )

    def _create_justification_tool(self) -> FunctionTool:
        """Инструмент создания обоснований"""

        # ✅ Захватываем self.llm в замыкание
        llm = self.llm

        def create_detailed_justification(user_query: str,
                                          movie_data: dict,
                                          evaluation_data: dict,
                                          **kwargs) -> dict:
            """Create detailed 4-5 sentence justification"""

            try:
                # распаковка
                if "title" not in movie_data and isinstance(movie_data, dict):
                    movie_data = dict(next(iter(movie_data.values())))
                movie_title = movie_data.get("title", "Unknown")
                overview = movie_data.get("overview", "")[:220]
                genres = movie_data.get("genres", "Unknown")
                vote_average = movie_data.get('vote_average', 0)
                imdb_rating = movie_data.get('imdb_rating', 0)

                justification_prompt = f"""
                You are a seasoned film critic. Write an ENGLISH explanation (exactly 4-5
sentences, one blank line, then a signature line).

USER QUERY:
"{user_query}"

MOVIE DATA
MOVIE DATA
Title          : {movie_title}
Genres         : {genres}
Overview (cut) : {overview}
TMDB / IMDb    : {vote_average}/10 • {imdb_rating}/10
RelevanceScore : {evaluation_data.get('final_score', 0)}

WRITING RULES
1. Output **only the finished justification**.  
2. NO planning words like "Next", "Then", "Need to", "Make sure", etc.  
3. NO meta-instructions or bullet lists.  
4. 1st-4th sentences must cover:
   • direct plot / theme overlap  
   • genre & narrative alignment  
   • one unique shared element  
   • (optionally) quality note via rating  
5. After a single blank line add EXACTLY:

"The relevance level of the film {movie_title} to your description is {evaluation_data.get('final_score', 0)}"
"""

                # ✅ Используем захваченную переменную llm
                response = llm.complete(justification_prompt)
                justification_text = response.text.strip()

                return {
                    "movie_title": movie_title,
                    "justification": justification_text,
                    "evaluation_score": evaluation_data.get('final_score', 0),
                    "quality_notes": self._extract_quality_notes(vote_average, imdb_rating),
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z"
                }

            except Exception as e:
                return {
                    "movie_title": movie_data.get('title', 'Unknown'),
                    "justification": f"Error creating justification: {str(e)}",
                    "error": str(e)
                }

        return FunctionTool.from_defaults(
            fn=create_detailed_justification,
            name="create_justification",
            description="Create detailed justification for movie recommendation"
        )

    @staticmethod
    def _extract_quality_notes(vote_average, imdb_rating):
        """Извлечение заметок о качестве"""
        notes = []
        if vote_average >= 8.0:
            notes.append("Высокий рейтинг TMDB")
        if imdb_rating >= 8.0:
            notes.append("Высокий рейтинг IMDb")
        if vote_average >= 7.0 and imdb_rating >= 7.0:
            notes.append("Стабильно высокие оценки")
        return notes

    @staticmethod
    def _trim_movie_data(movie_data: dict) -> dict:
        wanted = {"id", "title", "narrative_features"}
        return {k: movie_data.get(k) for k in wanted}

    # --- публичный прокси ---
    @classmethod
    def trim_movie_data(cls, movie_data: dict) -> dict:
        """Public wrapper for _trim_movie_data()."""
        return cls._trim_movie_data(movie_data)

        # -----------------------------------------------------------------------
        #  Анализ без ReAct

    def analyze_and_recommend(
            self,
            user_query: str,
            search_results: list[dict],
    ) -> dict:
        """
        1. Для каждого найденного фильма вызываем comprehensive_evaluation
        2. Выбираем top-3 через select_top3
        3. Для каждого из top-3 генерируем уникальный justification
        4. Формируем красивый card-output и возвращаем результат
        """

        logger.info("⏳ Start expert analysis for %d search results", len(search_results))

        # 1.  Комплексная оценка всех фильмов
        evaluated: list[dict] = []
        eval_tool = self.tools_map["comprehensive_evaluation"].fn  # короткая ссылка

        for item in search_results:
            movie_data: dict = item.get("movie_data", {})

            # ── включаем semantic_score / narrative_similarity внутрь movie_data
            movie_data = dict(movie_data)  # copy
            movie_data["semantic_score"] = item.get("semantic_score", 0)
            movie_data["narrative_similarity"] = item.get("narrative_similarity", 0)

            # ── вызов инструмента
            eval_result = eval_tool(user_query, movie_data)
            eval_result["movie_data"] = movie_data  # сохраняем для дальнейшего шага
            evaluated.append(eval_result)

        # 2.  Выбор top-3
        top3_res = self.tools_map["select_top3"].fn(evaluated)
        top3 = top3_res.get("top3_movies", [])
        logger.info("🏆 TOP-3 chosen: %s", [m["movie_title"] for m in top3])

        # 3.  Генерация обоснований
        just_tool = self.tools_map["create_justification"].fn
        cards, top3_details = [], []

        for idx, ev in enumerate(top3, 1):
            md = ev.pop("movie_data")  # достаём оригинальный movie_data
            just = just_tool(user_query, md, ev)["justification"]

            card = self._format_movie_card(md, just, idx)
            cards.append(card)

            top3_details.append(
                {
                    "rank": idx,
                    "movie_data": md,
                    "evaluation": ev,
                    "justification": just,
                }
            )

        # 4.  Финальный ответ
        return {
            "selected_movies": top3_details,
            "explanations": "\n\n---\n\n".join(cards),
            "analysis_complete": True,
            "methodology": "Direct Python calls (eval → top3 → justification)",
            "evaluated_at": datetime.datetime.utcnow().isoformat() + "Z",
        }

    def _create_unique_fallback_explanations(self, user_query: str, movies: list) -> str:
        """Создание уникальных fallback объяснений"""
        explanations = []

        for i, movie_result in enumerate(movies, 1):
            movie_data = movie_result.get('movie_data', {})
            title = movie_data.get('title', f'Movie {i}')

            # Генерируем уникальное обоснование даже в fallback режиме
            try:
                justification = self._create_simple_justification(user_query, movie_data, movie_result)
            except:
                justification = self._create_fallback_specific_justification(user_query, movie_data, movie_result)

            movie_card = self._format_movie_card(movie_data, justification, i)
            explanations.append(movie_card)

        return "\n\n---\n\n".join(explanations)

    def _extract_comprehensive_recommendations(self, response_text: str, user_query: str, search_results: list) -> dict:
        """Извлечение комплексных рекомендаций из ответа агента"""

        # Выбираем топ-3 фильма для обработки
        top_movies = search_results[:3]

        formatted_explanations = []
        top3_details = []
        movie_title = ''

        for i, movie_result in enumerate(top_movies, 1):
            movie_data = movie_result.get('movie_data', {})
            movie_title = movie_data.get('title', f'{i}: {movie_title}')

            # ✅ Всегда генерируем уникальное обоснование через LLM
            logger.info(f"Creating unique justification for movie {i}: {movie_title}")

            try:
                justification = self._create_simple_justification(user_query, movie_data, movie_result)

                # Проверка на generic ответы
                if len(justification) < 50 or "strong alignment" in justification.lower():
                    logger.warning(f"Generic justification detected for {movie_title}, retrying...")
                    # Повторная попытка с упрощенным промптом
                    justification = self._create_fallback_specific_justification(user_query, movie_data, movie_result)

            except Exception as e:
                logger.error(f"Error creating justification for {movie_title}: {e}")
                justification = self._create_fallback_specific_justification(user_query, movie_data, movie_result)

            # Форматирование для Gradio
            movie_card = self._format_movie_card(movie_data, justification, i)
            formatted_explanations.append(movie_card)

            # Детали для дальнейшей обработки
            top3_details.append({
                "rank": i,
                "movie_data": movie_data,
                "justification": justification,
                "evaluation_score": movie_result.get('final_score',
                                                     (movie_result.get('semantic_score', 0) + movie_result.get(
                                                         'narrative_similarity', 0)) / 2)
            })

        return {
            "top3_details": top3_details,
            "formatted_explanations": "\n\n---\n\n".join(formatted_explanations)
        }

    @staticmethod
    def _create_fallback_specific_justification(user_query: str, movie_data: dict, movie_result: dict) -> str:
        """Создание специфичного fallback обоснования"""

        title = movie_data.get('title', 'Unknown')
        genres = movie_data.get('genres', 'Unknown')
        overview = movie_data.get('overview', '')
        vote_avg = movie_data.get('vote_average', 0)
        semantic_score = movie_result.get('semantic_score', 0.0)

        # Извлекаем ключевые элементы из overview для специфичности
        overview_excerpt = overview[:150] + "..." if len(overview) > 150 else overview

        # Создаем более специфичное обоснование на основе доступных данных
        if vote_avg >= 8.0:
            quality_note = f"This critically acclaimed film (rated {vote_avg}/10)"
        elif vote_avg >= 7.0:
            quality_note = f"This well-received {genres.lower()} film"
        else:
            quality_note = f"This {genres.lower()} film"

        # Пытаемся выделить ключевые элементы сюжета
        plot_keywords = []
        user_lower = user_query.lower()
        overview_lower = overview.lower()

        # Простое совпадение ключевых слов для специфичности
        common_themes = []
        theme_words = ['love', 'war', 'family', 'revenge', 'friendship', 'betrayal', 'mystery',
                       'adventure', 'conflict', 'journey', 'discovery', 'redemption', 'survival']

        for theme in theme_words:
            if theme in user_lower and theme in overview_lower:
                common_themes.append(theme)

        themes_text = f"shared themes of {', '.join(common_themes[:2])}" if common_themes else "thematic parallels"

        return (f'{quality_note} "{title}" presents compelling connections to your story through {themes_text}. '
                f'The narrative elements in this film - {overview_excerpt} - create meaningful resonance with '
                f'the character dynamics and plot structure you\'ve described. With a semantic similarity score '
                f'of {semantic_score:.3f}, the film\'s approach to storytelling and conflict resolution aligns '
                f'well with the tone and themes present in your query.')

    def _create_simple_justification(self, user_query: str, movie_data: dict, movie_result: dict) -> str:
        """Создание простого обоснования"""

        # ✅ ПРЕДВАРИТЕЛЬНАЯ ИНИЦИАЛИЗАЦИЯ всех переменных
        title = movie_data.get('title', 'Unknown Movie')
        genres = movie_data.get('genres', 'Unknown')
        overview = movie_data.get('overview', 'No overview available')
        semantic_score = movie_result.get('semantic_score', 0.0)
        narrative_similarity = movie_result.get('narrative_similarity', 0.0)
        vote_avg = movie_data.get('vote_average', 0)
        imdb_rating = movie_data.get('imdb_rating', 0)
        release_year = str(movie_data.get('release_date', ''))[:4] \
            if movie_data.get('release_date') else 'Unknown'

        try:
            # overview = movie_data.get('overview', '')[:200] + "..."
            # ✅ Более детальный промпт для уникальных обоснований
            prompt = f"""
            You are an expert film critic. Create a unique, detailed 4-5 sentence 
            justification explaining why "{title}" specifically matches the user's query.

            User Query: "{user_query}"
            
            MOVIE DETAILS:
            - Title: {title} ({release_year})
            - Genres: {genres}
            - Overview: {overview}
            - TMDB Rating: {vote_avg}/10
            - IMDb Rating: {imdb_rating}/10
            - Semantic Match Score: {semantic_score:.3f}
            - Narrative Similarity: {narrative_similarity:.3f}

    REQUIREMENTS:
    1. Start by analyzing the SPECIFIC plot elements from this movie's overview that connect to the user's query
    2. Identify unique thematic parallels between the user's description and this particular film
    3. Explain how the genre and narrative structure align with what the user is seeking
    4. Mention specific character dynamics or plot mechanisms that match the user's story
    5. If ratings are high (7.0+), reference the quality as validation of the recommendation
    
    IMPORTANT: 
    - Be SPECIFIC to this movie - mention plot details, character types, conflicts from the overview
    - DO NOT use generic phrases like "demonstrates strong alignment" or "thematic elements"
    - Reference actual story elements that create the connection
    - Make each sentence build a specific argument for THIS movie
    - Focus on WHY this particular film matches, not general similarities
    
    Write as a knowledgeable film expert explaining a thoughtful recommendation.
    Ignore specific names and locations.
            """

            response = self.llm.complete(prompt)
            justification = response.text.strip()
            # return response.text.strip()

            # ✅ Проверка качества ответа - если слишком короткий или generic, попробуем еще раз
            if len(justification) < 100 or "demonstrates strong alignment" in justification.lower():
                # Второй более конкретный промпт
                retry_prompt = f"""
                Analyze specifically why "{title}" matches this user query: "{user_query}"
    
                Movie overview: {overview}
    
                Write 4-5 sentences explaining the specific connections:
                - What plot elements from "{title}" mirror the user's story?
                - How do the character conflicts in "{title}" relate to the user's description?
                - What genre-specific elements make this a good match?
                - Why would someone who likes the user's description enjoy "{title}"?
    
                Be concrete and specific about THIS movie's story elements.
                """

                retry_response = self.llm.complete(retry_prompt)
                justification = retry_response.text.strip()

            logger.info(f"Generated unique justification for {title}: {justification[:100]}...")
            return justification

        except (AttributeError, KeyError) as e:
            # ✅ Переменные гарантированно определены
            logger.warning(f"Data access error for {title}: {e}")
            # ✅ Улучшенный fallback с большей спецификой
            return (
                f'"{title}" ({release_year}) offers compelling parallels to your story through its {genres.lower()} '
                f'framework and narrative structure. The film\'s central conflicts and character dynamics '
                f'mirror key themes in your description, while its {semantic_score:.2f} semantic match score '
                f'indicates strong thematic alignment. The {vote_avg}/10 rating suggests quality storytelling '
                f'that would appeal to fans of your described plot elements.')

        except (ValueError, TypeError) as e:
            # ✅ Обработка ошибок преобразования данных
            logger.warning(f"Data processing error: {e}")
            return (f'This {genres} film "{title}" resonates with your query through its '
                    f'exploration of similar themes and narrative approaches. The story\'s '
                    f'structure and character development patterns '
                    f'create meaningful connections to your described plot, supported by '
                    f'a {semantic_score:.2f} '
                    f'matching score that indicates substantial thematic overlap.')

    @staticmethod
    def _safe_year(release_date) -> str:
        """Возвращает год как строку независимо от типа release_date."""
        from datetime import date, datetime

        if isinstance(release_date, (date, datetime)):
            return str(release_date.year)
        if isinstance(release_date, str) and len(release_date) >= 4:
            return release_date[:4]
        return "Unknown"

    @staticmethod
    def _format_movie_card(movie_data: dict, justification: str, rank: int) -> str:
        """Форматирование карточки фильма для Gradio"""

        # распаковка вложенного словаря
        if "title" not in movie_data and isinstance(movie_data, dict):
            movie_data = dict(next(iter(movie_data.values())))

        title = movie_data.get('title', 'Unknown')
        original_title = movie_data.get('original_title', '')
        release_date = movie_data.get('release_date', 'Unknown')
        year = ExpertAgent._safe_year(release_date)

        overview = movie_data.get('overview', 'No overview available')
        genres = movie_data.get('genres', 'Unknown')
        tagline = movie_data.get('tagline', '')

        vote_average = movie_data.get('vote_average', 0)
        vote_count = movie_data.get('vote_count', 0)
        imdb_rating = movie_data.get('imdb_rating', 0)
        popularity = movie_data.get('popularity', 0)

        runtime = movie_data.get('runtime', 0)
        budget = movie_data.get('budget', 0)
        revenue = movie_data.get('revenue', 0)

        director = movie_data.get('director', 'Unknown')
        cast = movie_data.get('cast', 'Unknown')

        # Форматирование карточки
        card = f"""**{rank}. {title}** ({year})
*{original_title}* {f'• {tagline}' if tagline else ''}

**Genres:** {genres}

**Overview:** {overview}

**📊 Ratings:**
• ⭐ TMDB: {vote_average}/10 ({vote_count:,} голосов)
• 🎬 IMDb: {imdb_rating}/10
• 📈 Popularity: {popularity:.0f}

**🎥 Technical data:**
• ⏱️ Runtime: {runtime} мин
• 💰 Budget: ${budget:,} USD
• 💵 Revenue: ${revenue:,} USD

**👥 Cast:**
• 🎬 Director: {director}
• 🎭 Cast: {cast}

**🎯 Justification:**
{justification}"""

        return card

    @staticmethod
    def _create_fallback_explanations(user_query: str, movies: list) -> str:
        """Создание fallback объяснений"""
        explanations = []

        for i, movie_result in enumerate(movies, 1):
            movie_data = movie_result.get('movie_data', {})
            title = movie_data.get('title', f'Movie {i}')

            explanation = (f"**{i}. {title}**\n\nThis film demonstrates strong alignment "
                           f"with your query through its thematic elements and narrative structure. "
                           f"The recommendation is based on semantic similarity analysis and genre "
                           f"matching.")
            explanations.append(explanation)

        return "\n\n---\n\n".join(explanations)
