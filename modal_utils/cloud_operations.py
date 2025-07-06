import re
import spacy

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import os
import tempfile


def clean_text(text):
    """Очищает текст от лишних символов и форматирования"""
    if pd.isna(text):
        return ""

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Замена спецсимволов на пробелы
    text = re.sub(r'\s+', ' ', text)       # Замена множественных пробелов на один
    return text.strip().lower()


def remove_entities(text, nlp):
    """Удаляет именованные сущности, стоп-слова и приводит к леммам"""
    if pd.isna(text) or text == "":
        return ""

    # nlp = spacy.load("en_core_web_lg")

    doc = nlp(text)

    # Удаляем именованные сущности, знаки пунктуации и стоп-слова,
    # применяем лемматизацию к оставшимся токенам
    tokens = [token.lemma_ for token in doc
              if token.ent_type_ not in ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'TIME', 'MONEY']
              and not token.is_punct
              and not token.is_stop
              and token.lemma_.strip()]

    return " ".join(tokens)


def prepare_text_for_embedding(text, nlp):
    """Полный процесс подготовки текста для эмбеддинга"""
    if pd.isna(text):
        return ""

    # Сначала очищаем текст
    cleaned_text = clean_text(text)

    # Затем удаляем сущности и выполняем лемматизацию
    processed_text = remove_entities(cleaned_text, nlp)

    return processed_text


def encode_user_query_fallback(query_text: str, device: str):
    """
    Fallback функция для случаев, когда spaCy недоступна
    """
    import re

    print("Fallback режим: обработка без spaCy")
    # Инициализация модели SentenceTransformer
    model = SentenceTransformer(
        'all-MiniLM-L6-v2',
        device=device,
        cache_folder="/tmp/model_cache"
    )

    if torch.cuda.is_available():
        model = model.half()

    # Простая очистка без spaCy
    processed_query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query_text)
    processed_query = re.sub(r'\s+', ' ', processed_query).strip().lower()

    # Простое удаление потенциальных имен (заглавные буквы)
    processed_query = re.sub(r'\b[A-Z][a-z]+\b', '', processed_query)
    processed_query = re.sub(r'\s+', ' ', processed_query).strip()

    print(f"Обработанный запрос: {processed_query}")

    # Генерация эмбеддинга
    query_embedding = model.encode(
        [processed_query],
        convert_to_tensor=False,
        batch_size=1,
        show_progress_bar=False
    )[0]

    # Базовые нарративные признаки без spaCy
    narrative_features = {
        "conflict_keywords": [],
        "plot_turns": 0,
        "sentiment_variance": 0.0,
        "action_density": 0.0
    }

    return {
        "original_query": query_text,
        "processed_query": processed_query,
        "embedding": query_embedding.tolist(),
        "embedding_dimension": len(query_embedding),
        "narrative_features": narrative_features,
        "device_used": device,
        "preprocessing_applied": True,
        "fallback_mode": True
    }


def heavy_computation(df=None, batch_size=128):
    """
    Основная функция обработки, поддерживает два режима:
    - Локальный: получает данные через параметр df
    - Облачный: читает данные из Volume (/data/input.csv)
    """
    # Определяем режим выполнения
    is_cloud_mode = df is None
    print(f'is_cloud_mode = {is_cloud_mode}')
    # 1. Загрузка данных (разные источники для локального и облачного режимов)
    if is_cloud_mode:
        # Облачный режим: читаем из Volume
        try:
            input_path = "/data/input.csv"
            if not os.path.exists(input_path):
                return {"error": "input_file_not_found"}

            df = pd.read_csv(input_path)
            print(f"Loaded {len(df)} movies from Volume")
        except Exception as e:
            return {"error": f"data_load_error: {str(e)}"}
    else:
        # Локальный режим: используем переданные данные
        print(f"Processing {len(df)} movies locally")

    # 2. Инициализация модели
    try:
        # Определяем кэш-директорию в зависимости от ОС
        cache_dir = os.path.join(tempfile.gettempdir(), "sentence_transformers_cache")

        # Создаем директорию, если не существует
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using cache directory: {cache_dir}")

        model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device="cuda" if torch.cuda.is_available() else "cpu",
            cache_folder=cache_dir
        )

        if torch.cuda.is_available():
            model = model.half()
            print("Using half-precision model")
    except Exception as e:
        return {"error": f"model_init_error: {str(e)}"}

    # 3. Создание эмбеддингов
    try:
        embeddings = []
        non_empty_overviews = df['processed_overview'].fillna("")

        # Если нет GPU, уменьшаем размер батча
        if not torch.cuda.is_available() and batch_size > 32:
            batch_size = 32
            print(f"Reduced batch_size to {batch_size} for CPU mode")

        for i in tqdm(range(0, len(non_empty_overviews), batch_size),
                      total=len(non_empty_overviews) // batch_size + 1):
            batch = non_empty_overviews.iloc[i:i + batch_size].tolist()
            batch_embeddings = model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.append(batch_embeddings.astype(np.float32))

        # 4. Сохранение результатов
        df['processed_overview_embedding'] = np.vstack(embeddings).tolist()
        # df['title_length'] = df['title'].apply(len)
        # df['has_overview'] = df['overview'].notna()

        # Для локального режима просто возвращаем результат
        if not is_cloud_mode:
            return {
                "status": "success",
                "processed": len(df),
                "embedding_dim": embeddings[0].shape[1] if embeddings else 0,
                "sample": {
                    "title": df['title'].iloc[0],
                    "overview": df['overview'].iloc[0][:50] + "..." if len(df['overview'].iloc[0]) > 50 else df['overview'].iloc[0],
                    "embedding_first_5": df['overview_embedding'].iloc[0][:5]
    }
            }

        # 5. Для облачного режима сохраняем в Volume
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
            df.to_csv(tmp, index=False)
            tmp_path = tmp.name

        # Копируем в Volume
        output_path = "/data/data/output.csv"
        with open(tmp_path, "rb") as src, open(output_path, "wb") as dst:
            dst.write(src.read())

        os.unlink(tmp_path)

        return {
            "status": "success",
            "processed": len(df),
            "embedding_dim": embeddings[0].shape[1] if embeddings else 0,
            "saved_path": output_path
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "type": type(e).__name__
        }


def parse_embedding_safe(embedding_data):
    """
    Оптимизированная функция парсинга эмбеддингов для формата строкового Python списка
    """
    import ast

    # Случай 1: Уже numpy array
    if isinstance(embedding_data, np.ndarray):
        return embedding_data

    # Случай 2: Python список
    if isinstance(embedding_data, list):
        return np.array(embedding_data, dtype=np.float32)

    # Случай 3: Строковое представление Python списка
    if isinstance(embedding_data, str):
        try:
            # Используем ast.literal_eval для безопасного парсинга
            parsed_list = ast.literal_eval(embedding_data.strip())
            if isinstance(parsed_list, list):
                return np.array(parsed_list, dtype=np.float32)
        except (ValueError, SyntaxError) as e:
            print(f"Ошибка парсинга эмбеддинга: {e}")
            return None

    return None


def extract_narrative_features_consistent(original_text: str, processed_text: str, nlp):
    """
    Извлечение нарративных признаков, идентичных тем, что используются для фильмов
    """
    from textacy.extract import keyterms
    from textblob import TextBlob
    import numpy as np
    import json

    # Инициализация структуры признаков (как в базе данных)
    features = {
        "conflict_keywords": [],
        "plot_turns": 0,
        "sentiment_variance": 0.0,
        "action_density": 0.0
    }

    try:
        # Обработка текста через spaCy
        processed_doc = nlp(processed_text) if processed_text and len(processed_text) >= 20 else None

        # 1. Ключевые слова конфликта (идентичный алгоритм)
        if processed_doc and len(processed_text) >= 20:
            try:
                conflict_terms = [
                    term for term, score in keyterms.textrank(
                        processed_doc,
                        topn=5,
                        window_size=10,
                        edge_weighting="count",
                        position_bias=False
                    ) if term and term.strip()
                ]
                features["conflict_keywords"] = conflict_terms
            except Exception as e:
                print(f"Ошибка извлечения ключевых слов: {e}")
                features["conflict_keywords"] = []

        # 2. Повороты сюжета (идентичный алгоритм)
        if original_text and len(original_text) >= 20:
            turn_keywords = {"but", "however", "though", "although", "nevertheless",
                             "suddenly", "unexpectedly", "surprisingly", "abruptly"}

            lower_text = original_text.lower()
            plot_turns_count = sum(lower_text.count(kw) for kw in turn_keywords)
            features["plot_turns"] = plot_turns_count

        # 3. Вариативность эмоций (идентичный алгоритм)
        if original_text and len(original_text) >= 20:
            try:
                blob = TextBlob(original_text)
                if len(blob.sentences) > 1:
                    sentiments = [s.sentiment.polarity for s in blob.sentences]
                    features["sentiment_variance"] = float(np.var(sentiments))
                else:
                    features["sentiment_variance"] = 0.0
            except Exception as e:
                print(f"Ошибка анализа эмоций: {e}")
                features["sentiment_variance"] = 0.0

        # 4. Плотность действий (идентичный алгоритм)
        if processed_doc and len(processed_doc) > 0:
            action_verbs = sum(1 for token in processed_doc if token.pos_ == "VERB")
            features["action_density"] = action_verbs / len(processed_doc)

    except Exception as e:
        print(f"Ошибка извлечения нарративных признаков: {e}")

    return features


def calculate_narrative_similarity(query_features, movie_features):
    """
    Вычисление сходства между нарративными признаками запроса и фильма.
    Использует те же 4 признака, что и в базе данных
    """
    import json

    if not movie_features or not query_features:
        return 0.0

    try:
        # Парсинг нарративных признаков фильма из JSON
        if isinstance(movie_features, str):
            movie_features_dict = json.loads(movie_features)
        else:
            movie_features_dict = movie_features

        # query_features уже является словарем
        query_features_dict = query_features

        # Вычисление сходства по каждому компоненту
        similarities = {}

        # 1. Сходство ключевых слов конфликта (Jaccard similarity)
        query_keywords = set(query_features_dict.get("conflict_keywords", []))
        movie_keywords = set(movie_features_dict.get("conflict_keywords", []))

        if query_keywords or movie_keywords:
            intersection = len(query_keywords.intersection(movie_keywords))
            union = len(query_keywords.union(movie_keywords))
            similarities["keywords"] = intersection / union if union > 0 else 0.0
        else:
            similarities["keywords"] = 0.0

        # 2. Сходство поворотов сюжета (нормализованная разность)
        query_turns = query_features_dict.get("plot_turns", 0)
        movie_turns = movie_features_dict.get("plot_turns", 0)
        max_turns = max(query_turns, movie_turns, 1)  # Избегаем деления на 0
        similarities["plot_turns"] = 1.0 - abs(query_turns - movie_turns) / max_turns

        # 3. Сходство эмоциональной вариативности
        query_sentiment_var = query_features_dict.get("sentiment_variance", 0.0)
        movie_sentiment_var = movie_features_dict.get("sentiment_variance", 0.0)
        max_sentiment_var = max(query_sentiment_var, movie_sentiment_var, 0.1)
        similarities["sentiment"] = 1.0 - abs(query_sentiment_var - movie_sentiment_var) / max_sentiment_var

        # 4. Сходство плотности действий
        query_action = query_features_dict.get("action_density", 0.0)
        movie_action = movie_features_dict.get("action_density", 0.0)
        max_action = max(query_action, movie_action, 0.1)
        similarities["action"] = 1.0 - abs(query_action - movie_action) / max_action

        # Взвешенная комбинация сходств
        weights = {
            "keywords": 0.4,  # Наибольший вес для ключевых слов
            "plot_turns": 0.25,  # Повороты сюжета важны
            "sentiment": 0.2,  # Эмоциональная окраска
            "action": 0.15  # Плотность действий
        }

        weighted_similarity = sum(
            similarities[key] * weights[key]
            for key in similarities.keys()
        )

        return weighted_similarity

    except Exception as e:
        print(f"Ошибка вычисления нарративного сходства: {e}")
        return 0.0


def rerank_by_narrative_features(candidates):
    """
    Переранжирование кандидатов с учетом нарративных признаков
    """
    for candidate in candidates:
        movie_data = candidate['movie_data']

        # Базовый семантический скор
        semantic_score = candidate['semantic_score']
        narrative_score = candidate.get('narrative_similarity', 0.0)

        # Веса для различных компонентов
        semantic_weight = 0.65      # Основной вес на семантику
        narrative_weight = 0.25     # Нарративные признаки
        quality_weight = 0.1        # Качественные метрики

        # Бонусы за качественные метрики
        rating_bonus = min(movie_data.get('vote_average', 0) / 10, 0.1)
        popularity_bonus = min(np.log(movie_data.get('popularity', 1)) / 10, 0.1)

        # Итоговый скор
        candidate['final_score'] = (
                semantic_score * semantic_weight +
                narrative_score * narrative_weight +
                rating_bonus + popularity_bonus
        )

        candidate['score_breakdown'] = {
            'semantic': semantic_score,
            'narrative': narrative_score,
            'quality': rating_bonus,
            'final': candidate['final_score']
        }

    # Сортировка по итоговому скору
    return sorted(candidates, key=lambda x: x['final_score'], reverse=True)
