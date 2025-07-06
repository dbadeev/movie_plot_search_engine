# agents/nebius_simple.py

import os
from openai import OpenAI
from typing import Any, Optional
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import Field, PrivateAttr


class SimpleNebiusLLM(CustomLLM):
    """Простой Nebius LLM через прямой OpenAI клиент"""

    # ✅ ИСПРАВЛЕНО: Объявляем поля как Pydantic поля
    api_key: str = Field(description="API ключ Nebius")
    model_name: str = Field(default="meta-llama/Llama-3.3-70B-Instruct", description="Имя модели")
    temperature: float = Field(default=0.3, description="Температура генерации")
    base_url: str = Field(default="https://api.studio.nebius.com/v1/", description="Base URL для API")

    # ✅ ИСПРАВЛЕНО: Используем PrivateAttr для OpenAI клиента
    _client: OpenAI = PrivateAttr()

    def __init__(
            self,
            api_key: str = None,
            model: str = "meta-llama/Llama-3.3-70B-Instruct",
            temperature: float = 0.3,
            **kwargs
    ):
        api_key = api_key or os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise ValueError("NEBIUS_API_KEY не найден")

        # ✅ ИСПРАВЛЕНО: Инициализируем родительский класс с полями
        super().__init__(
            api_key=api_key,
            model_name=model,
            temperature=temperature,
            **kwargs
        )

        # ✅ ИСПРАВЛЕНО: Создаем клиента как приватный атрибут
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=128000,
            num_output=4096,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Синхронное completion через Nebius API"""
        try:
            # ✅ ИСПРАВЛЕНО: Используем self._client
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=4096
            )

            text = response.choices[0].message.content
            return CompletionResponse(text=text)

        except Exception as e:
            error_text = f"Nebius API error: {str(e)}"
            return CompletionResponse(text=error_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        """Потоковое completion"""
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=4096,
                stream=True
            )

            accumulated_text = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    accumulated_text += content
                    yield CompletionResponse(text=accumulated_text, delta=content)

        except Exception as e:
            error_text = f"Nebius streaming error: {str(e)}"
            yield CompletionResponse(text=error_text, delta=error_text)


def create_nebius_llm(
        api_key: str = None,
        model: str = "meta-llama/Llama-3.3-70B-Instruct",
        temperature: float = 0.3
) -> SimpleNebiusLLM:
    """Фабричная функция для создания Nebius LLM"""
    return SimpleNebiusLLM(
        api_key=api_key,
        model=model,
        temperature=temperature
    )
