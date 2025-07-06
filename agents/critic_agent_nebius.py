# agents/critic_agent_nebius.py
from llama_index.core.agent import ReActAgent
# from llama_index.llms.openai import OpenAI
# from llama_index.llms.llama_api import LlamaAPI
from llama_index.core.tools import FunctionTool

from agents.nebius_simple import create_nebius_llm

import datetime
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FilmCriticAgent:
    def __init__(self, nebius_api_key: str):
        # ✅ Прямой Nebius LLM
        self.llm = create_nebius_llm(
            api_key=nebius_api_key,
            model="meta-llama/Llama-3.3-70B-Instruct-fast",
            # model="deepseek-ai/DeepSeek-R1-fast",
            temperature=0.7
        )

        self.tools = [
            self._create_overview_generation_tool(),
            self._create_overview_refinement_tool(),
            self._create_quality_assessment_tool()
        ]

        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            max_iterations=15,
            system_prompt=self._get_system_prompt()
        )

    @staticmethod
    def _get_system_prompt() -> str:
        """Статический метод для получения системного промпта.
        Обновленный системный промпт без технических префиксов"""

        return """You are a Film Critic Agent with expertise in movie analysis and synopsis writing.

Your responsibilities:
    1. Transform plot descriptions into professional movie overviews
    2. Ensure overviews match the style and structure of real movie descriptions
    3. Maintain narrative coherence and cinematic appeal
    4. Refine overviews based on user feedback

Use the Thought-Action-Observation cycle:
    - Think about the narrative elements and cinematic potential
    - Generate or refine the movie overview
    - Assess the quality and completeness
    - Make improvements until the overview meets professional standards
    - your final response should contain only the clean overview text without any technical formatting.

Write overviews in the style of IMDb or film database descriptions: 
engaging, informative, and capturing the essence of the story.

IMPORTANT OUTPUT RULES:
    - Generate ONLY the overview text itself
    - Do NOT add prefixes like "The final movie overview is:" or "Overview:"
    - Do NOT add explanatory text or comments
    - Write directly in the style of IMDb movie descriptions
    - Use present tense and engaging language
"""

    # @staticmethod
    def _create_overview_generation_tool(self) -> FunctionTool:
        """Статический метод для создания инструмента генерации overview"""

        def generate_movie_overview(plot_description: str) -> dict:
            """Generate a professional movie overview from a plot description using LLM"""

            # Placeholder для демонстрации структуры
            # generated_overview = f"A compelling story about {plot_description.lower()[:50]}..."
            try:
                prompt = f"""
                Transform this plot description into a professional movie overview similar to those on IMDb:

                Plot: "{plot_description}"

                Requirements:
                    - 80-200 words
                    - Engaging opening line
                    - Include main character roles (invent names if needed)
                    - Describe central conflict without spoilers
                    - Professional cinematic tone
                    - Present tense narration

                Generate ONLY the overview text, no additional commentary.
                
                CRITICAL: Return ONLY the overview text itself. 
                Do not add prefixes like "Movie overview:" or "The final overview is:". 
                Write directly as if this text will appear in a movie database.

                Example style: "When rookie detective Sarah Mitchell discovers a series of 
                mysterious disappearances in downtown Chicago, she uncovers a conspiracy that reaches 
                the highest levels of city government..."
                """

                response = self.llm.complete(prompt)
                generated_overview = response.text.strip()

                # # ✅ ДОПОЛНИТЕЛЬНАЯ ОЧИСТКА на уровне инструмента
                # cleaned_overview = self._clean_technical_prefixes(generated_overview)

                # Валидация длины
                word_count = len(generated_overview.split())
                quality_score = self._calculate_overview_quality(generated_overview)

                return {
                    "overview": generated_overview,
                    # "word_count": len(generated_overview.split()),
                    "word_count": word_count,
                    "quality_score": quality_score,
                    "meets_requirements": 80 <= word_count <= 200,
                    # "style": "professional",
                    "generated_at": datetime.datetime.utcnow().isoformat() + "Z"
                }

            except Exception as e:
                return {
                    "overview": f"Error generating overview: {str(e)}",
                    "word_count": 0,
                    "quality_score": 0.0,
                    "meets_requirements": False,
                    "error": str(e)
                }

        return FunctionTool.from_defaults(
            fn=generate_movie_overview,
            name="generate_overview",
            description="Generate a professional movie overview from a plot description"
        )

    @staticmethod
    def _clean_technical_prefixes(text: str) -> str:
        """Удаление технических префиксов из текста overview"""
        import re

        prefixes_to_remove = [
            r'^the final movie overview is:\s*',
            r'^final movie overview:\s*',
            r'^movie overview:\s*',
            r'^overview:\s*',
            r'^the overview is:\s*',
            r'^generated overview:\s*',
            r'^here\'?s the overview:\s*',
            r'^here is the overview:\s*',
            r'^\*\*movie overview\*\*:\s*',
            r'^\*\*overview\*\*:\s*'
        ]

        cleaned = text.strip()

        # Удаляем каждый возможный префикс
        for pattern in prefixes_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Удаляем кавычки если overview взят в кавычки
        cleaned = cleaned.strip('"\'')

        return cleaned.strip()

    # @staticmethod
    def _create_overview_refinement_tool(self) -> FunctionTool:
        """Статический метод для создания инструмента доработки overview"""

        def refine_overview(current_overview: str, user_feedback: str) -> dict:
            """Refine an overview based on user feedback"""

            # LLM вызов для доработки пользовательского описания
            try:
                prompt = f"""
                Improve this movie overview based on the user's feedback:

                Current overview: "{current_overview}"
                User feedback: "{user_feedback}"

                Requirements:
                - Apply the user's suggestions while maintaining professional quality
                - Keep 80-200 words
                - Maintain cinematic style and present tense
                - Preserve the core plot while incorporating changes

                Provide ONLY the refined overview text.
                """

                response = self.llm.complete(prompt)
                refined_overview = response.text.strip()

                word_count = len(refined_overview.split())
                quality_score = self._calculate_overview_quality(refined_overview)

                return {
                    "refined_overview": refined_overview,
                    "word_count": word_count,
                    "meets_requirements": 80 <= word_count <= 200,
                    # "changes_made": f"Applied user feedback: {user_feedback}",
                    "changes_applied": True,
                    # "quality_score": 0.9,
                    "quality_score": quality_score,
                    "refined_at": datetime.datetime.utcnow().isoformat() + "Z"
                }

            except Exception as e:
                return {
                    "refined_overview": current_overview,
                    "word_count": len(current_overview.split()),
                    "quality_score": 0.0,
                    "meets_requirements": False,
                    "changes_applied": False,
                    "error": str(e)
                }

        return FunctionTool.from_defaults(
            fn=refine_overview,
            name="refine_overview",
            description="Refine a movie overview based on user feedback"
        )

    @staticmethod
    def _create_quality_assessment_tool() -> FunctionTool:
        """Статический метод для создания инструмента оценки качества"""

        def assess_overview_quality(overview: str) -> dict:
            """Assess the quality of a movie overview"""

            words = overview.split()
            word_count = len(words)

            # Структурные проверки
            has_engaging_start = any(word in overview.lower() for word in
                                     ['when', 'after', 'as', 'in', 'during', 'follows', 'tells'])

            has_character_focus = any(word in overview.lower() for word in
                                      ['he', 'she', 'they', 'protagonist', 'character'])

            has_conflict = any(word in overview.lower() for word in
                               ['must', 'faces', 'discovers', 'confronts', 'struggles', 'battles'])

            # Проверка "кинематографичесого стиля (тона)"
            cinematic_words = ['journey', 'adventure', 'story', 'tale', 'epic', 'drama']
            has_cinematic_tone = any(word in overview.lower() for word in cinematic_words)

            # Итоговая оценка
            quality_factors = [
                80 <= word_count <= 200,
                has_engaging_start,
                has_character_focus,
                has_conflict,
                has_cinematic_tone
            ]

            # Базовая оценка качества
            # quality_score = 0.85
            quality_score = sum(quality_factors) / len(quality_factors)
            # structure_good = 80 <= word_count <= 200
            # engaging = any(
            #     word in overview.lower() for word in ['compelling', 'thrilling', 'captivating', 'intriguing'])

            return {
                "quality_score": quality_score,
                "word_count": word_count,
                "length_appropriate": 80 <= word_count <= 200,
                # "structure_good": structure_good,
                # "engaging": engaging,
                "has_engaging_start": has_engaging_start,
                # "needs_improvement": not structure_good or not engaging,
                "needs_improvement": quality_score < 0.7,
                "assessed_at": datetime.datetime.utcnow().isoformat() + "Z"
            }

        return FunctionTool.from_defaults(
            fn=assess_overview_quality,
            name="assess_quality",
            description="Assess the quality and completeness of a movie overview"
        )

    def create_overview(self, plot_description: str) -> dict:
        """Создание overview фильма с LLM вызовом(НЕ статический - использует self.agent)"""

        prompt = f"""
        Create a professional movie overview based on this plot description:

        "{plot_description}"

        Use your generate_overview tool to create an engaging overview that sounds like it belongs 
        in a movie database.
        Then assess the quality and refine if necessary.
        
        Remember: Output only the clean overview text, no prefixes or technical comments.
        """

        try:
            response = self.agent.chat(prompt)

            # Извлечение overview из ответа агента и дополнительная очистка
            overview_text = self._extract_overview_from_response(str(response))
            final_overview = self._clean_technical_prefixes(overview_text)

            return {
                "overview": final_overview,  # ✅ Дважды очищенный текст
                "status": "generated",
                "ready_for_search": True,
                "generated_at": datetime.datetime.utcnow().isoformat() + "Z"
            }

        except Exception as e:
            return {
                "overview": f"Error creating overview: {str(e)}",
                "status": "error",
                "ready_for_search": False,
                "error": str(e)
            }

        # return self._parse_overview_response(response)

    def refine_with_feedback(self, overview: str, feedback: str) -> dict:
        """Доработка overview на основе обратной связи (НЕ статический - использует self.agent)"""
        prompt = f"""
        Please refine this movie overview based on the user's feedback:

        Current overview: "{overview}"
        User feedback: "{feedback}"

        Use your refine_overview tool to make the necessary improvements while maintaining professional quality.
        """

        try:
            response = self.agent.chat(prompt)
            refined_overview = self._extract_overview_from_response(str(response))

            return {
                "overview": refined_overview,
                "status": "refined",
                "ready_for_search": True,
                "refined_at": datetime.datetime.utcnow().isoformat() + "Z"
            }

        except Exception as e:
            return {
                "overview": overview,  # Возврат оригинала при ошибке
                "status": "error",
                "ready_for_search": True,
                "error": str(e)
            }

        # return self._parse_overview_response(response)

    @staticmethod
    def _extract_overview_from_response(response_text: str) -> str:
        """Извлечение чистого overview из ответа агента с удалением технических префиксов"""
        import re

        # Список технических префиксов, которые нужно удалить
        technical_prefixes = [
            "The final movie overview is:",
            "Final movie overview:",
            "Movie overview:",
            "Overview:",
            "The overview is:",
            "Generated overview:",
            "Here's the overview:",
            "Here is the overview:",
            "The generated overview:",
            "Movie description:",
            "Film overview:",
            "**Movie Overview:**",
            "**Overview:**"
        ]

        # Поиск текста, который выглядит как overview
        lines = response_text.split('\n')
        candidate_lines = [line.strip() for line in lines if len(line.strip().split()) > 20]

        best_overview = ""

        if candidate_lines:
            # Берем самую длинную содержательную строку
            best_candidate = max(candidate_lines, key=len)

            # ✅ УДАЛЕНИЕ ТЕХНИЧЕСКИХ ПРЕФИКСОВ
            cleaned_text = best_candidate.strip()

            # Удаляем известные префиксы (регистронезависимо)
            for prefix in technical_prefixes:
                # Создаем паттерн для поиска префикса в начале строки
                pattern = rf'^{re.escape(prefix)}\s*'
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

            # Удаляем кавычки, если overview взят в кавычки
            cleaned_text = cleaned_text.strip('"\'')

            # Удаляем возможные markdown форматирования
            cleaned_text = re.sub(r'^#+\s*', '', cleaned_text)  # Заголовки
            cleaned_text = re.sub(r'^\*\*.*?\*\*\s*', '', cleaned_text)  # Жирный текст

            best_overview = cleaned_text.strip()

        # Fallback: очистка всего текста от технических деталей
        if not best_overview or len(best_overview.split()) < 20:
            # Удаляем технические части ReAct агента
            fallback_text = re.sub(r'Tool:|Thought:|Action:|Observation:', '', response_text)

            # Удаляем все известные префиксы
            for prefix in technical_prefixes:
                pattern = rf'{re.escape(prefix)}\s*'
                fallback_text = re.sub(pattern, '', fallback_text, flags=re.IGNORECASE)

            best_overview = fallback_text.strip()

        return best_overview

    @staticmethod
    def _calculate_overview_quality(overview: str) -> float:
        """Расчет качества overview"""
        words = overview.split()
        word_count = len(words)

        quality_factors = []

        # Длина
        if 80 <= word_count <= 200:
            quality_factors.append(1.0)
        else:
            quality_factors.append(max(0.5, 1.0 - abs(word_count - 100) / 100))

        # Наличие ключевых элементов
        has_plot_elements = any(word in overview.lower() for word in
                                ['story', 'follows', 'discovers', 'must', 'when', 'after'])
        quality_factors.append(1.0 if has_plot_elements else 0.5)

        # Отсутствие спойлеров
        no_spoilers = not any(word in overview.lower() for word in
                              ['ending', 'dies', 'kills', 'twist', 'revealed'])
        quality_factors.append(1.0 if no_spoilers else 0.7)

        return sum(quality_factors) / len(quality_factors)

    # @staticmethod
    # def _parse_overview_response(response) -> dict:
    #     """Статический метод для парсинга ответа агента"""
    #     import datetime
    #
    #     return {
    #         "overview": str(response),
    #         "status": "generated",
    #         "ready_for_search": True,
    #         "generated_at": datetime.datetime.utcnow().isoformat() + "Z"
    #     }
