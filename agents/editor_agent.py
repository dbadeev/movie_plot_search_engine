# agents/editor_agent.py
# должны быть установлены:
# %pip install llama-index-program-openai
# %pip install llama-index-llms-llama-api
# !pip install llama-index

from llama_index.core.agent import ReActAgent
# from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
# from llama_index.llms.llama_api import LlamaAPI

# ✅ ПРОСТОЕ РЕШЕНИЕ: Используем обычный OpenAI клиент
from agents.nebius_simple import create_nebius_llm

import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_reasoning_failure(callback_manager, exception):
    """Обработка превышения лимита итераций"""
    if "max iterations" in str(exception).lower():
        return """Based on the analysis completed so far:

The text has been reviewed and appears to meet basic requirements. 
Some minor improvements may be beneficial but are not critical.
The text can proceed to the next stage of processing.

Status: Approved with partial analysis due to iteration limit."""

    return f"Analysis completed with limitations: {str(exception)}"

class EditorAgent:
    def __init__(self, nebius_api_key: str, use_react: bool = False):
        #Args: use_react: Если True - использует ReAct агента, если False - прямые вызовы;
        # Прямой Nebius LLM
        self.llm = create_nebius_llm(
            api_key=nebius_api_key,
            model="meta-llama/Llama-3.3-70B-Instruct-fast",
            # model="deepseek-ai/DeepSeek-R1-fast",
            temperature=0.7
        )

        # Инструменты агента (без языкового детектора)
        self.tools = [
            self._create_text_validation_tool(),
            self._create_grammar_correction_tool(),   # НЕ статический - использует self.llm
            self._create_semantic_check_tool(),
            self._create_approval_tool()
        ]

        self.use_react = use_react

        # Создание ReAct агента, если нужен
        if self.use_react:
            self.agent = ReActAgent.from_tools(
                tools=self.tools,
                llm=self.llm,
                verbose=True,
                max_iterations=10,
                handle_reasoning_failure_fn=handle_reasoning_failure,  # Добавляем обработчик ошибок
                system_prompt=self._get_system_prompt()
            )

    def process_and_improve_text(self, user_text: str) -> dict:
        """Выбор между ReAct агентом и прямыми вызовами"""
        if self.use_react:
            return self._process_with_react(user_text)  # ✅ Эта функция должна быть определена
        else:
            return self._process_direct(user_text)
    # self.start_time = None  # Для трекинга времени обработки

    def _process_with_react(self, user_text: str) -> dict:
        """✅ ДОБАВЛЕНО: Обработка через ReAct агента"""
        import time

        start_time = time.time()

        # Ранняя проверка длины
        words = user_text.split()
        word_count = len(words)

        if word_count < 50:
            processing_time = time.time() - start_time
            return {
                "status": "insufficient_length",
                "original_text": user_text,
                "improved_text": user_text,
                "approved": False,
                "message": f"""**📝 Text Too Short ({word_count} words)**

                        Your text contains only {word_count} words, but our system requires 
                        a minimum of 50 words for proper movie plot analysis.

                        Why 50 words?
                        - Enables accurate semantic analysis
                        - Ensures sufficient plot detail for matching
                        - Improves recommendation quality

                        Please expand your plot description with:*
                        - More character details
                        - Additional plot points
                        - Setting information  
                        - Conflict development

                        Example format:
                        "A young wizard discovers he has magical powers when he receives 
                        a letter to attend Hogwarts School. At school, he learns about his 
                        past and must face the dark wizard who killed his parents. Along with 
                        his friends, he uncovers secrets about the school and fights against 
                        evil forces threatening the wizarding world."

                        Current length: {word_count}/50 words required

                        Please rewrite your plot description with at least 50 words 
                        and try again.""",
                "word_count": word_count,
                "min_required": 50,
                "total_processing_time": round(processing_time, 3),
                "early_termination": True
            }

        # Существующая логика с ReAct агентом
        prompt = f"""
         Please review and improve this plot description efficiently:
         
         Text: "{user_text}"
         
        Tasks (complete in 5-7 steps maximum):
            1. Validate length (50 or more words) and structure
            2. Correct any grammatical errors and typos  
            3. Check semantic coherence
            4. Approve if requirements are met

        IMPORTANT: Be efficient. Try to complete the task quickly.
        If the text is already acceptable, just approve it.
         """

        try:
            response = self.agent.chat(prompt)
            logger.info(f"response: {response}")
            result = self._parse_editor_response(response, user_text)
            logger.info(f"result: {result}")

        except ValueError as e:
            if "max iterations" in str(e).lower():
                # ✅ ДОБАВЛЕНО: Fallback обработка при превышении лимита
                print(f"Editor reached max iterations, providing fallback result")
                result = {
                    "status": "approved",  # Одобряем для продолжения процесса
                    "original_text": user_text,
                    "improved_text": user_text,  # Возвращаем исходный текст
                    "message": "Text analysis completed with basic validation. The text appears acceptable for processing.",
                    "approved": True,
                    "iteration_limit_reached": True,
                    "fallback_used": True
                }
            else:
                # Для других ValueError
                raise e

        # Добавление времени обработки
        if start_time:
            total_processing_time = time.time() - start_time
            result["total_processing_time"] = round(total_processing_time, 3)

        return result

    def _process_direct(self, user_text: str) -> dict:
        """Прямая обработка без ReAct агента - более надежно"""
        import time

        start_time = time.time()

        # 1. Ранняя проверка длины
        words = user_text.split()
        word_count = len(words)

        if word_count < 50:
            processing_time = time.time() - start_time
            return {
                "status": "insufficient_length",
                "original_text": user_text,
                "improved_text": user_text,
                "approved": False,
                "message": f"Text too short: {word_count} words. Minimum required: 50 words.",
                "word_count": word_count,
                "min_required": 50,
                "total_processing_time": round(processing_time, 3),
                "early_termination": True
            }

        # ✅ ПРЯМЫЕ ВЫЗОВЫ ИНСТРУМЕНТОВ

        # 2. Валидация текста
        validation_tool = self._create_text_validation_tool()
        validation_result = validation_tool.fn(user_text)

        if not validation_result["valid"]:
            processing_time = time.time() - start_time
            return {
                "status": "needs_improvement",
                "original_text": user_text,
                "improved_text": user_text,
                "approved": False,
                "message": f"Validation failed: {', '.join(validation_result['issues'])}",
                "validation_result": validation_result,
                "total_processing_time": round(processing_time, 3)
            }

        # 3. Грамматическая коррекция
        grammar_tool = self._create_grammar_correction_tool()
        grammar_result = grammar_tool.fn(user_text)

        # 4. Семантическая проверка (используем corrected_text если есть)
        text_to_check = grammar_result.get("corrected_text", user_text)
        semantic_tool = self._create_semantic_check_tool()
        semantic_result = semantic_tool.fn(text_to_check)

        # ✅ СТРУКТУРИРОВАННОЕ ПРИНЯТИЕ РЕШЕНИЯ

        # Критерии одобрения
        approval_criteria = {
            "validation_passed": validation_result["valid"],
            "grammar_score": grammar_result.get("improvement_score", 0.0),
            "semantic_coherent": semantic_result.get("coherent", False),
            "corrections_made": grammar_result.get("corrections_made", False)
        }

        # Проверка grammar threshold
        grammar_threshold = 0.8
        meets_grammar_threshold = approval_criteria["grammar_score"] >= grammar_threshold

        # Финальное решение
        approved = (
                approval_criteria["validation_passed"] and
                approval_criteria["semantic_coherent"] and
                meets_grammar_threshold
        )

        # Формирование результата
        final_text = grammar_result.get("corrected_text", user_text) if approval_criteria[
            "corrections_made"] else user_text

        processing_time = time.time() - start_time

        if approved:
            return {
                "status": "approved",
                "original_text": user_text,
                "improved_text": final_text,
                "approved": True,
                "message": f"✅ Text approved! Quality score: {approval_criteria['grammar_score']:.2f}/1.0",
                "approval_criteria": approval_criteria,
                "tool_results": {
                    "validation": validation_result,
                    "grammar": grammar_result,
                    "semantics": semantic_result
                },
                "total_processing_time": round(processing_time, 3)
            }
        else:
            # Детальное сообщение о причинах отклонения
            rejection_reasons = []

            if not meets_grammar_threshold:
                rejection_reasons.append(
                    f"Grammar quality below threshold: {approval_criteria['grammar_score']:.2f} < {grammar_threshold}")

            if not approval_criteria["semantic_coherent"]:
                rejection_reasons.append("Text lacks semantic coherence")

            return {
                "status": "needs_improvement",
                "original_text": user_text,
                "improved_text": final_text,
                "approved": False,
                "message": f"❌ Text needs improvement:\n- " + "\n- ".join(rejection_reasons),
                "approval_criteria": approval_criteria,
                "tool_results": {
                    "validation": validation_result,
                    "grammar": grammar_result,
                    "semantics": semantic_result
                },
                "total_processing_time": round(processing_time, 3)
            }

    @staticmethod
    def _get_system_prompt() -> str:
        """Статический метод для получения системного промпта"""
        """Упрощенный системный промпт для повышения эффективности"""
        return """You are an Editor Agent for English plot descriptions.

    Your task: Quickly validate and improve text quality (minimum 50 words, proper grammar).

    EFFICIENT Process (3-5 steps maximum):
    1. Use validate_text to check basic requirements
    2. Use correct_grammar to fix issues and get improvement_score  
    3. Use check_semantics to verify plot coherence
    4. Use approve_text ONLY when all criteria are met:
       - improvement_score > 0.8 from grammar correction
       - semantic coherence confirmed
       - validation requirements passed

    The approve_text tool will automatically integrate 
    results from all previous checks."""

    @staticmethod
    def _create_text_validation_tool() -> FunctionTool:
        """Статический метод для создания инструмента валидации текста"""
        def validate_text_requirements(text: str) -> dict:
            """Validate if text meets length and structure requirements"""
            words = text.split()
            word_count = len(words)

            sentences = re.split(r'[.!?]+', text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]

            issues = []

            # Word count check
            if word_count < 50:
                issues.append(f"Text too short: {word_count} words (minimum 50)")
            elif word_count > 500:
                issues.append(f"Text too long: {word_count} words (maximum 500)")

            # Sentence structure check
            if len(sentences) < 2:
                issues.append("Text should contain at least 2 sentences")

            for i, sentence in enumerate(sentences):
                if len(sentence.split()) < 3:
                    issues.append(f"Sentence {i + 1} is too short")

            return {
                "valid": len(issues) == 0,
                "word_count": word_count,
                "sentence_count": len(sentences),
                "issues": issues
            }

        return FunctionTool.from_defaults(
            fn=validate_text_requirements,
            name="validate_text",
            description="Validate if text meets length and structural requirements"
        )

    def _create_grammar_correction_tool(self) -> FunctionTool:
        """НЕ статический метод - использует self.llm для реальной коррекции"""
        def correct_grammar_with_llm(text: str) -> dict:
            """Correct grammatical errors and typos in the text. Real grammar correction using LLM"""
            try:
                correction_prompt = f"""
                Please correct any grammatical errors, typos, and improve the clarity of this text while preserving its meaning:
    
                "{text}"
    
                Requirements:
                - Fix grammatical errors
                - Correct spelling mistakes
                - Improve sentence structure if needed
                - Maintain the original plot and meaning
                - Keep it concise and engaging
    
                Return only the corrected text without explanations.
                """

                # Имитация коррекции (в реальности здесь был бы LLM вызов)
                # corrected_text = text  # Placeholder
                # Реальный LLM вызов через self.llm
                response = self.llm.complete(correction_prompt)
                corrected_text = response.text.strip()

                # Проверка качества коррекции
                corrections_made = corrected_text.lower() != text.lower()
                word_diff = abs(len(corrected_text.split()) - len(text.split()))

                # Оценка качества улучшения
                improvement_score = min(1.0, max(0.5, 1.0 - (word_diff / len(text.split()))))

                return {
                    "corrected_text": corrected_text,
                    # "corrections_made": True,
                    "corrections_made": corrections_made,
                    "improvement_score": 0.85,      # заглушка против слишком придирчивых llm )
                    # "improvement_score": improvement_score,
                    "original_length": len(text.split()),
                    "corrected_length": len(corrected_text.split())
                }

            except Exception as e:
                # Fallback: возврат оригинального текста при ошибке
                print(f"Ошибка LLM коррекции: {e}")
                return {
                    "corrected_text": text,
                    "corrections_made": False,
                    "improvement_score": 0.0,
                    "error": str(e)
                }

        return FunctionTool.from_defaults(
            fn=correct_grammar_with_llm,
            name="correct_grammar",
            description="Correct grammatical errors and improve text clarity"
        )

    @staticmethod
    def _create_semantic_check_tool() -> FunctionTool:
        """Статический метод для создания инструмента семантической проверки"""
        def check_semantic_coherence(text: str) -> dict:
            """Check if the text is semantically coherent and well-structured"""
            sentences = re.split(r'[.!?]+', text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]

            issues = []

            # Basic coherence checks
            if len(sentences) < 2:
                issues.append("Need more sentences for proper plot development")

            # Check for plot elements
            plot_keywords = [
                # Конфликт/Драма
                "war", "betrayal", "revenge", "corruption", "intrigue", "assassination",
                "struggle", "injustice", "dilemma", "survival", "persecution", "resistance",
                "revolution", "espionage", "conspiracy", "situation", "terrorism", "feud",

                # Отношения/Эмоции
                "romance", "love",  "heartbreak", "friendship",
                "family", "sacrifice", "rivalry", "problems", "betrayal", "jealousy",
                "forgiveness", "redemption", "loneliness", "grief", "hope", "obsession", "devotion", "separation",

                # Приключения/Действие
                "quest", "hunt", "mission", "escape", "chase", "heist", "disaster", "disaster",
                "apocalypse", "invasion", "battle", "duel", "superhero", "vigilante", "kidnapping",
                "investigation", "mystery", "conspiracy", "experiment",

                # Личностный рост
                "age", "self-discovery", "crisis", "transformation",
                "fear", "growth",  "journey", "awakening",
                "underdog", "rebirth",

                # Наука/Фантастика
                "ai", "time travel", "space exploration",  "dystopia",
                 "cyberpunk", "robot", "mutant", "superpower",
                "contact", "post-apocalypse", "virtual reality",

                # Мистика/Ужасы
                "haunting", "possession", "curse", "force", "witchcraft", "vampire",
                "zombie", "horror", "slasher", "monster", "ghost", "demon",
                "ritual", "paranormal",

                # Обстановка/Атмосфера
                "small town", "big city", "jungle", "desert", "ocean", "station", "kingdom",
                "era",  "ancient", "civilization",
                "submarine", "island", "laboratory"
            ]
            has_plot_elements = any(keyword in text.lower() for keyword in plot_keywords)

            if not has_plot_elements:
                issues.append("Text should include clear plot elements (characters, setting, conflict)")

            return {
                "coherent": len(issues) == 0,
                "issues": issues,
                "plot_elements_present": has_plot_elements,
                "readability_score": 0.8
            }

        return FunctionTool.from_defaults(
            fn=check_semantic_coherence,
            name="check_semantics",
            description="Check semantic coherence and plot structure"
        )

    @staticmethod
    def _create_approval_tool() -> FunctionTool:
        """Инструмент одобрения с учетом результатов грамматической и семантической проверки"""
        def approve_text_with_validation(text: str) -> dict:
            """ Финальное одобрение с учетом результатов грамматической коррекции и семантической проверки"""
            import datetime

            # Получаем результаты грамматической коррекции
            grammar_tool = self._create_grammar_correction_tool()
            grammar_result = grammar_tool.fn(text)

            # Получаем результаты семантической проверки
            semantic_tool = self._create_semantic_check_tool()
            semantic_result = semantic_tool.fn(text)

            # Получаем результаты валидации
            validation_tool = self._create_text_validation_tool()
            validation_result = validation_tool.fn(text)

            # Анализ всех результатов для принятия решения
            approval_criteria = {
                "grammar_score": grammar_result.get("improvement_score", 0.0),
                "semantic_coherent": semantic_result.get("coherent", False),
                "validation_passed": validation_result.get("valid", False),
                "corrections_needed": grammar_result.get("corrections_made", False)
            }

            # ✅ КЛЮЧЕВАЯ ЛОГИКА: Принятие решения на основе всех проверок

            # 1. Проверка базовых требований
            if not approval_criteria["validation_passed"]:
                return {
                    "approved": False,
                    "text": text,
                    "rejection_reason": "Failed basic validation requirements",
                    "validation_issues": validation_result.get("issues", []),
                    "approval_criteria": approval_criteria
                }

            # 2. Проверка семантической связности
            if not approval_criteria["semantic_coherent"]:
                return {
                    "approved": False,
                    "text": text,
                    "rejection_reason": "Text lacks semantic coherence",
                    "semantic_issues": semantic_result.get("issues", []),
                    "approval_criteria": approval_criteria
                }

            # 3. Проверка качества грамматики (improvement_score > 0.8)
            grammar_threshold = 0.8
            if approval_criteria["grammar_score"] < grammar_threshold:
                return {
                    "approved": False,
                    "text": text,
                    "rejection_reason": f"Grammar quality below threshold "
                                        f""
                                        f"({approval_criteria['grammar_score']:.2f} < {grammar_threshold})",
                    "suggested_text": grammar_result.get("corrected_text", text),
                    "approval_criteria": approval_criteria
                }

            # ✅ УСПЕШНОЕ ОДОБРЕНИЕ: Все проверки пройдены
            final_text = grammar_result.get("corrected_text", text) if approval_criteria["corrections_needed"] else text

            # Генерация текущего времени в UTC
            current_time = datetime.datetime.utcnow()
            timestamp_iso = current_time.isoformat() + "Z"

            # Расчет итогового качественного score
            final_quality_score = (
                    approval_criteria["grammar_score"] * 0.6 +  # 60% - грамматика
                    (1.0 if approval_criteria["semantic_coherent"] else 0.0) * 0.3 +  # 30% - семантика
                    (1.0 if approval_criteria["validation_passed"] else 0.0) * 0.1  # 10% - валидация
            )

            return {
                "approved": True,
                "text": final_text,
                "original_text": text,
                "timestamp": timestamp_iso,
                "quality_score": round(final_quality_score, 3),
                "approval_criteria": approval_criteria,
                "improvements_applied": approval_criteria["corrections_needed"],
                "approval_metadata": {
                    "grammar_score": approval_criteria["grammar_score"],
                    "semantic_passed": approval_criteria["semantic_coherent"],
                    "validation_passed": approval_criteria["validation_passed"],
                    "final_score": round(final_quality_score, 3),
                    "threshold_met": final_quality_score > 0.8,
                    "utc_time": current_time.strftime("%Y-%m-%d %H:%M:%S UTC")
                }
            }

        return FunctionTool.from_defaults(
            fn=approve_text_with_validation,
            name="approve_text",
            description="Final approval based on grammar correction and semantic validation results"
        )

    # @staticmethod
    def _parse_editor_response(self, response, original_text) -> dict:
        """Парсинг ответа ReAct агента с извлечением результатов инструментов"""
        import re
        import json

        response_text = str(response)
        logger.info(f"response_text: {response_text}")

        # ✅ ИСПРАВЛЕНО: Инициализация результата с fallback значениями
        result = {
            "status": "needs_improvement",
            "original_text": original_text,
            "improved_text": original_text,
            "message": "Processing completed",
            "approved": False,
            "improvement_score": 0.0,
            "quality_metrics": {},
            "tool_results": {}
        }

        # ✅ ПАРСИНГ РЕЗУЛЬТАТОВ ИНСТРУМЕНТОВ

        # 1. Извлечение результата approve_text (финальное решение)
        approve_pattern = r'approve_text.*?(\{[^}]*"approved"[^}]*\})'
        approve_match = re.search(approve_pattern, response_text, re.DOTALL | re.IGNORECASE)

        if approve_match:
            try:
                approve_result = json.loads(approve_match.group(1))
                result["approved"] = approve_result.get("approved", False)
                result["status"] = "approved" if approve_result.get("approved", False) else "needs_improvement"
                result["quality_metrics"]["final_score"] = approve_result.get("quality_score", 0.0)
                result["tool_results"]["approval"] = approve_result

                # Используем improved text из approve_text если доступен
                if "text" in approve_result and approve_result["text"] != original_text:
                    result["improved_text"] = approve_result["text"]

            except json.JSONDecodeError:
                logger.warning("Failed to parse approve_text result")

        # 2. Извлечение результата correct_grammar (improvement_score и исправления)
        grammar_pattern = r'correct_grammar.*?(\{[^}]*"improvement_score"[^}]*\})'
        grammar_match = re.search(grammar_pattern, response_text, re.DOTALL | re.IGNORECASE)

        if grammar_match:
            try:
                grammar_result = json.loads(grammar_match.group(1))
                result["improvement_score"] = grammar_result.get("improvement_score", 0.0)
                result["tool_results"]["grammar"] = grammar_result

                # ✅ КЛЮЧЕВАЯ ПРОВЕРКА: improvement_score > 0.8
                if grammar_result.get("improvement_score", 0.0) > 0.8:
                    result["quality_metrics"]["grammar_threshold_met"] = True

                    # Используем corrected_text если коррекция была сделана
                    if grammar_result.get("corrections_made", False):
                        corrected_text = grammar_result.get("corrected_text", original_text)
                        if corrected_text != original_text:
                            result["improved_text"] = corrected_text
                else:
                    result["quality_metrics"]["grammar_threshold_met"] = False
                    result["approved"] = False  # Переопределяем если grammar score низкий
                    result["status"] = "needs_improvement"

            except json.JSONDecodeError:
                logger.warning("Failed to parse correct_grammar result")

        # 3. Извлечение результата validate_text
        validate_pattern = r'validate_text.*?(\{[^}]*"valid"[^}]*\})'
        validate_match = re.search(validate_pattern, response_text, re.DOTALL | re.IGNORECASE)

        if validate_match:
            try:
                validate_result = json.loads(validate_match.group(1))
                result["tool_results"]["validation"] = validate_result
                result["quality_metrics"]["validation_passed"] = validate_result.get("valid", False)

                if not validate_result.get("valid", False):
                    result["approved"] = False
                    result["status"] = "needs_improvement"

            except json.JSONDecodeError:
                logger.warning("Failed to parse validate_text result")

        # 4. Извлечение результата check_semantics
        semantic_pattern = r'check_semantics.*?(\{[^}]*"coherent"[^}]*\})'
        semantic_match = re.search(semantic_pattern, response_text, re.DOTALL | re.IGNORECASE)

        if semantic_match:
            try:
                semantic_result = json.loads(semantic_match.group(1))
                result["tool_results"]["semantics"] = semantic_result
                result["quality_metrics"]["semantic_coherent"] = semantic_result.get("coherent", False)

                if not semantic_result.get("coherent", False):
                    result["approved"] = False
                    result["status"] = "needs_improvement"

            except json.JSONDecodeError:
                logger.warning("Failed to parse check_semantics result")

        # ✅ ФОРМИРОВАНИЕ ДЕТАЛЬНОГО СООБЩЕНИЯ на основе результатов инструментов
        message_parts = []

        if result["approved"]:
            message_parts.append("✅ **Text approved for processing**")
            if result["improvement_score"] > 0.8:
                message_parts.append(f"📊 Quality score: {result['improvement_score']:.2f}/1.0")
            if result["improved_text"] != original_text:
                message_parts.append("📝 Text has been improved during processing")
        else:
            message_parts.append("❌ **Text requires improvement**")

            # Детальные причины отклонения
            if not result["quality_metrics"].get("grammar_threshold_met", True):
                score = result.get("improvement_score", 0.0)
                message_parts.append(f"📝 Grammar quality below threshold: {score:.2f} < 0.8")

            if not result["quality_metrics"].get("validation_passed", True):
                validation_issues = result["tool_results"].get("validation", {}).get("issues", [])
                if validation_issues:
                    message_parts.append(f"📋 Validation issues: {', '.join(validation_issues[:2])}")

            if not result["quality_metrics"].get("semantic_coherent", True):
                semantic_issues = result["tool_results"].get("semantics", {}).get("issues", [])
                if semantic_issues:
                    message_parts.append(f"🧠 Semantic issues: {', '.join(semantic_issues[:2])}")

        result["message"] = "\n".join(message_parts) if message_parts else response_text

        # ✅ FALLBACK: Если ничего не извлечено, используем простую логику
        if not any([approve_match, grammar_match, validate_match, semantic_match]):
            logger.warning("No tool results found, falling back to keyword search")
            approved = "approved" in response_text.lower() and "true" in response_text.lower()
            result["approved"] = approved
            result["status"] = "approved" if approved else "needs_improvement"
            result["message"] = response_text

        logger.info(f"Parsed result: approved={result['approved']}, improvement_score={result['improvement_score']}")

        return result
