import modal
from modal import Image, App, Volume, Secret
import logging
# import modal
import faiss
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import torch
import os

from agents.modal_agents import app as agents_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image = (
    Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install(
        "build-essential",
        "python3-dev",
        "gcc",
        "g++",
        "cmake",
        "wget",  # Добавляем wget
        "unzip"  # Добавляем unzip для распаковки
    )
    .pip_install_from_requirements("requirements_modal.txt")
    .add_local_file(
        local_path="setup_image.py",
        remote_path="/root/setup_image.py",
        copy=True
    )
    # ✅ ДОБАВЛЕНО: Добавляем скрипт извлечения punkt данных
    .add_local_file(
        local_path="setup_punkt_extraction.py",
        remote_path="/root/setup_punkt_extraction.py",
        copy=True
    )

    .run_commands("python /root/setup_image.py")
    .run_commands("python -m spacy download en_core_web_lg")


    .run_commands(
        # Скачиваем ресурс punkt
        "wget -O /tmp/punkt.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip",
        "unzip /tmp/punkt.zip -d /tmp",

        # Создаем структуру директорий
        "mkdir -p /root/nltk_data/tokenizers/punkt_tab/english",

        # Копируем основные файлы
        "cp /tmp/punkt/PY3/english.pickle /root/nltk_data/tokenizers/punkt_tab/english/",
        "cp /tmp/punkt/README /root/nltk_data/tokenizers/punkt_tab/",
        "cp -r /tmp/punkt/PY3 /root/nltk_data/tokenizers/punkt_tab/",

        # ✅ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Запускаем скрипт извлечения данных
        "python /root/setup_punkt_extraction.py",

        # Удаляем временные файлы
        "rm -rf /tmp/punkt*"
    )
    .add_local_dir("modal_utils", remote_path="/root/modal_utils")
    # .add_local_dir("local_utils", remote_path="/root/local_utils")
    .add_local_dir("agents", remote_path="/root/agents")
)

app = App(
    name="tmdb-project",
    image=image,
    secrets=[
        Secret.from_name("my-env"),  # Для конфиденциальных данных
        Secret.from_name("nebius-secret")
    ]
)

# Включаем все функции агентов в основной app
app.include(agents_app)

volume = Volume.from_name("tmdb-data", create_if_missing=True)


@app.function(
    volumes={"/data": volume},
    gpu="A10G",
    timeout=3600
)
def process_movies():
    """Основная функция обработки фильмов"""
    # Импорт внутри функции для работы с добавленными директориями
    from modal_utils.cloud_operations import heavy_computation
    return heavy_computation()


@app.function(volumes={"/data": volume})
def upload_file(data_str: str):
    import shutil
    import os

    volume.listdir(path='/', recursive=True)
    print(volume.listdir(path='/', recursive=True))

    # local_file_path = 'temp_sample.csv'  # Используйте временный файл
    remote_file_path = '/data/input.csv'  # Путь в Volume

    print(1)
    # Создаем директорию, если нужно
    os.makedirs(os.path.dirname(remote_file_path), exist_ok=True)

    # Записываем данные напрямую в файл
    with open(remote_file_path, "w") as f:
        f.write(data_str)

    print(f"Данные успешно записаны в Volume: {remote_file_path}")
    return remote_file_path


@app.function(
    image=image,
    gpu="A10G",  # было any
    volumes={"/data": volume},
    timeout=120  # было 1800 == 30 минут на батч
)
def process_batch(batch: list[tuple]):
    """
    Обрабатывает батч данных на GPU
    Вход: список кортежей (processed_text, original_text)
    Выход: список JSON-строк с признаками
    """
    import spacy
    from textacy.extract import keyterms
    from textblob import TextBlob
    import numpy as np
    import json
    import en_core_web_lg  # Прямой импорт модели
    import torch
    from concurrent.futures import ThreadPoolExecutor

    torch.set_num_threads(1)  # Уменьшаем число CPU потоков
    spacy.prefer_gpu()  # Активирует GPU для spaCy

    # Загружаем модель
    nlp = en_core_web_lg.load()

    # Добавляем sentencizer, если его нет в пайплайне
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    processed_texts = [item[0] for item in batch]
    original_texts = [item[1] for item in batch]

    # Обработка предобработанных текстов (Оптимизированная обработка spaCy)
    # processed_docs = list(nlp.pipe(processed_texts, batch_size=128))
    processed_docs = list(nlp.pipe(processed_texts, batch_size=4096))  # Увеличьте для GPU - было 400 для CPU

    # Функция для параллельного вычисления эмоциональной вариативности (sentiment variance)
    def compute_sentiment_variance(text):
        if not text or len(text) < 20:
            return 0.0

        try:
            blob = TextBlob(text)
            if len(blob.sentences) > 1:
                sentiments = [s.sentiment.polarity for s in blob.sentences]
                return float(np.var(sentiments))
            return 0.0
        except:
            return 0.0

    # Параллельное вычисление для всего батча
    with ThreadPoolExecutor(max_workers=16) as executor:
        sentiment_variances = list(executor.map(compute_sentiment_variance, original_texts))

    # Предварительно вычисляем plot_turns для всего батча
    turn_keywords = {"but", "however", "though", "although", "nevertheless",
                     "suddenly", "unexpectedly", "surprisingly", "abruptly"}

    # Векторизованный расчет plot_turns.
    # Используем более эффективный метод
    lower_texts = [text.lower() for text in original_texts]
    plot_turns_counts = [
        sum(text.count(kw) for kw in turn_keywords)
        if text and len(text) >= 20 else 0
        for text in lower_texts
    ]

    results = []
    for i, (processed_doc, original_text) in enumerate(zip(processed_docs, original_texts)):
        features = {
            "conflict_keywords": [],
            "plot_turns": plot_turns_counts[i],  # Используем предвычисленное значение (было 0, вычислялось позже)
            "sentiment_variance": sentiment_variances[i],  # Используем предвычисленное значение (было 0.0)
            "action_density": 0.0
        }

        try:
            # 1. Ключевые слова конфликта
            if processed_texts[i] and len(processed_texts[i]) >= 20:
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

            # Плотность действий
            if processed_doc and len(processed_doc) > 0:
                action_verbs = sum(1 for token in processed_doc if token.pos_ == "VERB")
                features["action_density"] = action_verbs / len(processed_doc)

        except Exception as e:
            print(f"Error processing item {i}: {str(e)[:100]}")

        results.append(json.dumps(features))

    return results


@app.function(
    image=image,
    volumes={"/data": volume},
    # memory=6144,  # Увеличиваем память до 6 ГБ
    timeout=600  # 150 минут вместо 60 секунд
)
def load_data(max_rows: int = None):
    """Загружает данные из CSV на Volume"""
    import pandas as pd

    # file_path = "/data/data/output.csv"
    file_path = "/data/data/output.parquet"  # Теперь используем Parquet
    print(f"Loading data from {file_path}...")

    # Чтение данных с возможностью ограничения количества строк
    if max_rows:
        # df = pd.read_csv(file_path, nrows=max_rows)
        df = pd.read_parquet(file_path, rows=max_rows)
    else:
        # Чтение всего файла
        df = pd.read_parquet(file_path)
        # df = pd.read_csv(file_path)

    print(f"Loaded {len(df)} records")

    # Проверка необходимых столбцов
    required_columns = ['processed_overview', 'overview']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")
    print(f'Columns check finished')

    # Заполнение пропущенных значений
    df['processed_overview'] = df['processed_overview'].fillna('')
    df['overview'] = df['overview'].fillna('')
    print(f'Missing values filling is finished')

    return df


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300  # 5 минут вместо 60 секунд
)
def save_results(df, output_path):
    """Сохраняет результаты на Volume"""
    print(f"Saving results to {output_path}...")
    # df.to_parquet(output_path, index=False)
    df.to_parquet(output_path, index=False, engine='pyarrow')  # или engine='fastparquet'
    print(f"✅ Results saved to {output_path}")


@app.local_entrypoint()
def process_test_batch(batch_size: int = 1000):
    """Обрабатывает тестовый батч из Volume"""
    import json
    # Загрузка данных
    df = load_data.remote(max_rows=batch_size)

    # Формирование батча
    batch_data = list(zip(
        df['processed_overview'].astype(str),
        df['overview'].astype(str)
    ))

    # Обработка батча
    print(f"Processing test batch ({len(batch_data)} records) on GPU...")
    results = process_batch.remote(batch_data)

    # Добавление результатов
    df['narrative_features'] = results
    df['features_decoded'] = df['narrative_features'].apply(json.loads)

    # Сохранение результатов
    output_path = f"/data/data/test_batch_results_{batch_size}.parquet"
    save_results.remote(df, output_path)

    # Вывод статистики
    print("\nProcessing statistics:")
    print(
        f"Conflict_keywords (non-empty): {sum(1 for x in df['features_decoded'] if x['conflict_keywords'])}/{len(df)}")
    print(f"Avg plot_turns: {df['features_decoded'].apply(lambda x: x['plot_turns']).mean():.2f}")
    print(f"Avg sentiment_variance: {df['features_decoded'].apply(lambda x: x['sentiment_variance']).mean():.4f}")
    print(f"Avg action_density: {df['features_decoded'].apply(lambda x: x['action_density']).mean():.2f}")

    print("\n✅ Test batch processing complete!")


# @app.local_entrypoint()
# def process_full_dataset(batch_size: int = 15000):
#     """Обрабатывает весь датасет на Volume"""
#     from tqdm import tqdm
#
#     # Загрузка данных
#     print(f'Loading data is started')
#     df = load_data.remote()
#     print(f'Loading data is finished')
#     total_records = len(df)
#     print(f"Processing full dataset: {total_records} records")
#
#     # Формирование батчей
#     batch_data = list(zip(
#         df['processed_overview'].astype(str),
#         df['overview'].astype(str)
#     ))
#
#     batches = [
#         batch_data[i:i + batch_size]
#         for i in range(0, total_records, batch_size)
#     ]
#
#     # Обработка батчей
#     all_results = []
#     for i, batch in enumerate(tqdm(batches, desc="Processing batches")):
#         print(f"Processing batch {i + 1}/{len(batches)} ({len(batch)} records)...")
#         results = process_batch.remote(batch)
#         all_results.extend(results)
#
#     # Добавление результатов
#     df['narrative_features'] = all_results
#
#     # Сохранение результатов
#     output_path = "/data/data/full_dataset_results.parquet"
#     save_results.remote(df, output_path)
#
#     print(f"\n✅ Full dataset processing complete! Results saved to {output_path}")
#     print(f"Total records processed: {len(df)}")


@app.local_entrypoint()
def show_sample_results(file_path: str = "/data/test_batch_results_1000.parquet"):
    """Показывает примеры результатов из файла на Volume"""
    import json

    # Загрузка результатов
    @app.function(volumes={"/data": volume})
    def load_results(path):
        import pandas as pd

        return pd.read_parquet(path)

    df = load_results.remote(file_path)

    # Добавление декодированных признаков
    if 'narrative_features' in df.columns:
        df['features_decoded'] = df['narrative_features'].apply(json.loads)

    print(f"Results from {file_path} ({len(df)} records):")

    # Вывод примеров
    sample_size = min(3, len(df))
    print(f"\nSample of {sample_size} records:")
    for i, row in df.head(sample_size).iterrows():
        print(f"\nRecord {i}:")
        print(f"Processed: {row['processed_overview'][:100]}...")
        print(f"Original: {row['overview'][:100]}...")
        print("Features:")
        features = row['features_decoded'] if 'features_decoded' in row else json.loads(row['narrative_features'])
        for k, v in features.items():
            print(f"  {k}: {v}")

    # Общая статистика
    if 'features_decoded' in df.columns:
        print("\nDataset statistics:")
        print(f"Avg plot_turns: {df['features_decoded'].apply(lambda x: x['plot_turns']).mean():.2f}")
        print(f"Avg sentiment_variance: {df['features_decoded'].apply(lambda x: x['sentiment_variance']).mean():.4f}")
        print(f"Avg action_density: {df['features_decoded'].apply(lambda x: x['action_density']).mean():.2f}")


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,  # 1 час на конвертацию
    memory=8192  # 8 ГБ памяти
)
def convert_csv_to_parquet():
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.csv as pc
    from pathlib import Path
    import time

    start_time = time.time()

    input_path = "/data/data/output.csv"
    output_path = "/data/data/output.parquet"

    print(f"Starting conversion: {input_path} -> {output_path}")

    # Создаем директорию если нужно
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Читаем CSV с помощью PyArrow (оптимизировано для больших файлов)
    reader = pc.open_csv(
        input_path,
        read_options=pc.ReadOptions(block_size=128 * 1024 * 1024),  # 128MB блоки
        parse_options=pc.ParseOptions(delimiter=",")
    )

    # Схема для записи Parquet
    writer = None

    # Обрабатываем данные порциями
    batch_count = 0
    while True:
        try:
            batch = reader.read_next_batch()
            if not batch:
                break

            df = batch.to_pandas()

            if writer is None:
                # Инициализируем writer при первом батче
                writer = pq.ParquetWriter(
                    output_path,
                    pa.Table.from_pandas(df).schema,
                    compression='SNAPPY'
                )

            # Конвертируем в pyarrow Table и записываем
            table = pa.Table.from_pandas(df)
            writer.write_table(table)

            batch_count += 1
            print(f"Processed batch {batch_count} ({df.shape[0]} rows)")

        except StopIteration:
            break

    # Финализируем запись
    if writer:
        writer.close()

    duration = time.time() - start_time
    print(f"✅ Conversion complete! Saved to {output_path}")
    print(f"Total batches: {batch_count}")
    print(f"Total time: {duration:.2f} seconds")

    return output_path


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600
)
def rebuild_parquet_with_row_index():
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_path = "/data/data/output.parquet"
    output_path = "/data/data/output_indexed.parquet"

    # Читаем исходные данные
    df = pd.read_parquet(input_path)

    # Добавляем индекс строки
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'row_id'}, inplace=True)

    # Сохраняем с разбивкой по строкам
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, row_group_size=15000)

    return output_path


@app.local_entrypoint()
def process_full_dataset(batch_size: int = 15000):
    """Обрабатывает и сохраняет результаты напрямую в Volume"""
    from tqdm import tqdm
    import math

    # 1. Получаем только метаданные (количество строк)
    total_records = get_row_count.remote()
    print(f"Total records to process: {total_records}")

    # 2. Рассчитываем количество батчей
    num_batches = math.ceil(total_records / batch_size)
    print(f"Processing in {num_batches} batches of {batch_size} records")

    # 3. Создаем временную директорию для частичных результатов
    partial_dir = "/data/partial_results"

    # 4. Обрабатываем и сохраняем результаты по батчам.
    # # Запускаем все батчи параллельно
    # calls = []
    # for batch_idx in range(num_batches):
    #     call = process_and_save_batch.spawn(
    #         start_row=batch_idx * batch_size,
    #         end_row=min((batch_idx + 1) * batch_size, total_records),
    #         batch_idx=batch_idx,
    #         partial_dir=partial_dir
    #     )
    #     calls.append(call)
    #
    # # Ожидаем завершения всех
    # for call in calls:
    #     call.get()
    # num_batches = 2

    # !!! Отработало, поэтому комментим !!!
    # for batch_idx in tqdm(range(num_batches)):
    #     start_row = batch_idx * batch_size
    #     end_row = min((batch_idx + 1) * batch_size, total_records)
    #
    #     # Запускаем обработку батча и сохранение
    #     process_and_save_batch.remote(
    #         start_row=start_row,
    #         end_row=end_row,
    #         batch_idx=batch_idx,
    #         partial_dir=partial_dir
    #     )

    # 5. Объединяем результаты
    final_path = "/data/data/full_dataset_results.parquet"
    combine_results.remote(partial_dir, final_path)

    print("\n✅ Full dataset processing complete!")
    print(f"Results saved to {final_path}")


@app.function(volumes={"/data": volume})
def init_partial_dir(partial_dir: str):
    """Создает директорию для частичных результатов"""
    import os
    os.makedirs(partial_dir, exist_ok=True)
    return f"Created directory {partial_dir}"


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    timeout=300
)
def process_and_save_batch(start_row: int, end_row: int, batch_idx: int, partial_dir: str):
    """Обрабатывает батч и сохраняет результаты в отдельный файл"""
    import pandas as pd
    import pyarrow.parquet as pq
    import os

    # 0. Создаем директорию, если ее нет
    os.makedirs(partial_dir, exist_ok=True)

    # 1. Чтение данных
    file_path = "/data/data/output.parquet"
    # parquet_file = pq.ParquetFile(file_path)
    # table = parquet_file.read_row_group(0).slice(start_row, end_row - start_row)
    # df = table.to_pandas()
    # Альтернативный метод чтения без row groups
    df = pd.read_parquet(file_path)
    df = df.iloc[start_row:end_row]

    # 2. Подготовка данных
    df['processed_overview'] = df['processed_overview'].fillna('')
    df['overview'] = df['overview'].fillna('')

    # 3. Формирование батча
    batch_data = list(zip(
        df['processed_overview'].astype(str),
        df['overview'].astype(str)
    ))

    # 4. Обработка батча
    results = process_batch.remote(batch_data)

    # 5. Сохранение результатов
    result_df = pd.DataFrame({'narrative_features': results})
    output_path = os.path.join(partial_dir, f"batch_{batch_idx}.parquet")
    result_df.to_parquet(output_path)

    return f"Saved batch {batch_idx} to {output_path}"


@app.function(volumes={"/data": volume})
def combine_results(partial_dir: str, final_path: str):
    """Объединяет частичные результаты в финальный файл"""
    import pandas as pd
    import os
    from glob import glob
    import pyarrow.parquet as pq

    # 1. Сбор всех частичных файлов
    # partial_files = glob(os.path.join(partial_dir, "*.parquet"))
    partial_files = sorted(
        glob(os.path.join(partial_dir, "*.parquet")),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )
    print(partial_files)
    # 2. Чтение и объединение
    full_results = []
    for file_path in partial_files:
        df = pd.read_parquet(file_path)
        full_results.extend(df['narrative_features'].tolist())

    print(f'len(full_results) = {len(full_results)}')
    # 3. Чтение исходных данных
    source_df = pd.read_parquet("/data/data/output.parquet")
    print({source_df.info()})

    # 4. Добавляем результаты
    source_df['narrative_features'] = full_results

    # 5. Сохранение финального результата
    source_df.to_parquet(final_path)

    # 6. Очистка временных файлов
    for file_path in partial_files:
        os.remove(file_path)
    os.rmdir(partial_dir)

    return f"Combined {len(partial_files)} batches into {final_path}"


@app.function(volumes={"/data": volume})
def get_row_count():
    """Возвращает общее количество строк в Parquet файле"""
    import pyarrow.parquet as pq

    file_path = "/data/data/output.parquet"
    return pq.read_metadata(file_path).num_rows


# import modal
# import faiss
# import numpy as np
# import pandas as pd
# import pickle
# from sentence_transformers import SentenceTransformer
# import torch
# import os

# app = modal.App("plotmatcher")
# volume = modal.Volume.from_name("plotmatcher-data", create_if_missing=True)

# Используем ваш существующий образ
# image = (
#     Image.from_registry(
#         "nvidia/cuda:12.8.1-devel-ubuntu22.04",
#         add_python="3.11"
#     )
#     .apt_install(
#         "build-essential",
#         "python3-dev",
#         "gcc",
#         "g++",
#         "cmake",
#         "wget",
#         "unzip"
#     )
#     .pip_install_from_requirements("requirements_modal.txt")
#     .add_local_file(
#         local_path="setup_image.py",
#         remote_path="/root/setup_image.py",
#         copy=True
#     )
#     .run_commands("python /root/setup_image.py")
#     .run_commands("python -m spacy download en_core_web_lg")
# )


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    timeout=3600,
    memory=16384
)
def build_faiss_index():
    """
    Построение FAISS индекса с учетом совместимости CUDA 12.8
        Исправленная версия для эмбеддингов в формате строкового Python списка
    """
    import ast

    print("Проверка доступности CUDA...")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA устройств: {torch.cuda.device_count()}")
        print(f"Текущее устройство: {torch.cuda.current_device()}")

    # Загрузка данных
    df = pd.read_parquet("/data/data/full_dataset_results.parquet")
    print(f"Загружено {len(df)} фильмов")

    # Анализ формата первого эмбеддинга
    sample_embedding = df['processed_overview_embedding'].iloc[0]
    print(f"Пример эмбеддинга: {str(sample_embedding)[:100]}...")
    print(f"Тип данных: {type(sample_embedding)}")

    # Извлечение эмбеддингов
    embeddings_list = []
    valid_indices = []
    parse_errors = 0

    print("Начинаем обработку эмбеддингов...")

    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            embedding_data = row['processed_overview_embedding']

            # Обработка различных форматов хранения эмбеддингов.
            # А именно - парсинг строкового представления Python списка
            if isinstance(embedding_data, str):
                try:
                    # Безопасный парсинг с помощью ast.literal_eval
                    parsed_list = ast.literal_eval(embedding_data.strip())
                    if isinstance(parsed_list, list):
                        embedding = np.array(parsed_list, dtype=np.float32)
                    else:
                        parse_errors += 1
                        continue
                except (ValueError, SyntaxError):
                    parse_errors += 1
                    continue
            elif isinstance(embedding_data, list):
                embedding = np.array(embedding_data, dtype=np.float32)
            elif isinstance(embedding_data, np.ndarray):
                embedding = embedding_data.astype(np.float32)
            else:
                parse_errors += 1
                continue

            # Проверка размерности (Размерность all-MiniLM-L6-v2 = 384)
            if len(embedding) == 384:
                embeddings_list.append(embedding.astype(np.float32))
                valid_indices.append(idx)
            else:
                parse_errors += 1

        except Exception as e:
            parse_errors += 1
            if parse_errors <= 5:  # Выводим первые несколько ошибок
                print(f"Ошибка обработки эмбеддинга {idx}: {e}")
            continue

        # Прогресс каждые 50000 записей
        if (idx + 1) % 50000 == 0:
            print(f"Обработано {idx + 1}/{len(df)} записей, валидных: {len(embeddings_list)}")

    print(f"Успешно обработано {len(embeddings_list)} эмбеддингов из {len(df)}")
    print(f"Ошибок парсинга: {parse_errors}")
    print(f"Успешность обработки: {len(embeddings_list) / len(df) * 100:.2f}%")

    if not embeddings_list:
        raise ValueError(f"Не найдено валидных эмбеддингов. Всего ошибок: {parse_errors}")

    # Создание матрицы эмбеддингов
    embeddings_matrix = np.vstack(embeddings_list)
    print(f"Подготовлено {len(embeddings_matrix)} эмбеддингов")
    print(f"Создана матрица эмбеддингов: {embeddings_matrix.shape}")

    # Нормализация для косинусного сходства
    faiss.normalize_L2(embeddings_matrix)
    print("Эмбеддинги нормализованы")

    # Создание FAISS индекса с поддержкой GPU
    dimension = embeddings_matrix.shape[1]

    # Проверяем доступность GPU для FAISS
    if faiss.get_num_gpus() > 0:
        print("Используем GPU для построения FAISS индекса")
        # GPU ресурсы
        res = faiss.StandardGpuResources()

        # CPU индекс
        cpu_index = faiss.IndexFlatIP(dimension)

        # Перенос на GPU
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(embeddings_matrix)

        # Возврат на CPU для сохранения
        index = faiss.index_gpu_to_cpu(gpu_index)
        print("FAISS индекс построен на GPU и перенесен на CPU для сохранения")
    else:
        print("Используем CPU для FAISS")
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_matrix)

    # Сохранение результатов
    print("Сохранение FAISS индекса...")
    faiss.write_index(index, "/data/data/movie_embeddings.index")

    # Сохранение метаданных
    print("Сохранение метаданных фильмов...")
    valid_movies_df = df.iloc[valid_indices].reset_index(drop=True)
    valid_movies_df.to_parquet("/data/data/indexed_movies_metadata.parquet")

    result = {
        "status": "success",
        "total_movies": len(valid_movies_df),
        "original_dataset_size": len(df),
        "index_size": index.ntotal,
        "dimension": dimension,
        "gpu_used": faiss.get_num_gpus() > 0,
        "processing_success_rate": len(valid_indices) / len(df),
        "parse_errors": parse_errors
    }

    print("=" * 50)
    print("ПОСТРОЕНИЕ ИНДЕКСА ЗАВЕРШЕНО")
    print(f"Обработано фильмов: {result['total_movies']} из {result['original_dataset_size']}")
    print(f"Размерность векторов: {result['dimension']}")
    print(f"Успешность: {result['processing_success_rate'] * 100:.2f}%")
    print("=" * 50)

    return result


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300
)
def test_embedding_parsing(num_samples=100):
    """
    Тестирование парсинга эмбеддингов на небольшой выборке данных
    """
    import ast

    df = pd.read_parquet("/data/data/full_dataset_results.parquet")
    print(f"Загружено {len(df)} фильмов для тестирования")

    test_sample = df.head(num_samples)
    successful_parses = 0
    failed_parses = 0

    print("Тестирование парсинга эмбеддингов...")

    for idx, row in test_sample.iterrows():
        embedding_data = row['processed_overview_embedding']

        try:
            if isinstance(embedding_data, str):
                parsed_list = ast.literal_eval(embedding_data.strip())
                if isinstance(parsed_list, list):
                    embedding = np.array(parsed_list, dtype=np.float32)
                    if len(embedding) == 384:
                        successful_parses += 1
                    else:
                        print(f"Неправильная размерность {len(embedding)} для индекса {idx}")
                        failed_parses += 1
                else:
                    print(f"Парсинг не дал список для индекса {idx}: {type(parsed_list)}")
                    failed_parses += 1
            else:
                print(f"Неожиданный тип данных для индекса {idx}: {type(embedding_data)}")
                failed_parses += 1

        except Exception as e:
            print(f"Ошибка парсинга для индекса {idx}: {e}")
            failed_parses += 1

    print(f"\nРезультаты тестирования:")
    print(f"Успешных парсингов: {successful_parses}")
    print(f"Неудачных парсингов: {failed_parses}")
    print(f"Успешность: {successful_parses / (successful_parses + failed_parses) * 100:.2f}%")

    return {
        "successful_parses": successful_parses,
        "failed_parses": failed_parses,
        "success_rate": successful_parses / (successful_parses + failed_parses)
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=300
)
def encode_user_query(query_text: str, remove_entities: bool = True):
    """
    Генерация эмбеддинга для пользовательского описания с опциональным удалением именованных сущностей
    """
    import spacy
    import tempfile
    # Импорт внутри функции для работы с добавленными директориями
    from modal_utils.cloud_operations import (clean_text, prepare_text_for_embedding,
                                              encode_user_query_fallback, extract_narrative_features_consistent)

    # Проверка входных данных
    if not query_text or not query_text.strip():
        raise ValueError("Пустой запрос не может быть обработан")

    # Определение устройства
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")

    # Инициализация spaCy модели (та же, что использовалась для обработки фильмов)
    try:
        try:
            # Загрузка spaCy с проверкой GPU
            if torch.cuda.is_available():
                spacy.prefer_gpu()

            import en_core_web_lg
            nlp = en_core_web_lg.load()
            # nlp = spacy.load("en_core_web_lg")

            # Добавляем sentencizer, если его нет
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")

            print("SpaCy модель загружена успешно")

        except Exception as e:
            print(f"Ошибка загрузки spaCy: {e}")
            # Fallback к простой обработке
            return encode_user_query_fallback(query_text, device)

        # Инициализация модели для кодирования
        try:
            # Определяем кэш-директорию в зависимости от ОС
            cache_dir = os.path.join(tempfile.gettempdir(), "sentence_transformer_cache")

            # Создаем директорию, если не существует
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Using cache directory: {cache_dir}")

            model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=device,
                cache_folder=cache_dir
            )

            # Оптимизация для GPU
            if torch.cuda.is_available():
                model = model.half()
                print("Using half-precision model")
        except Exception as e:
            return {"error": f"model_init_error: {str(e)}"}

        # Применяем тот же процесс обработки, что и для фильмов

        # Опциональное удаление именованных сущностей для фокуса на сюжете
        if remove_entities:
            processed_query = prepare_text_for_embedding(query_text, nlp)

            # Проверка, что после обработки остался текст
            if not processed_query.strip():
                print("Предупреждение: После удаления сущностей текст стал пустым, используем очищенную версию")
                processed_query = clean_text(query_text)
        else:
            processed_query = clean_text(query_text)

        # Финальная проверка
        if not processed_query.strip():
            processed_query = query_text.lower().strip()

        print(f"Исходное описание: '{query_text}'")
        print(f"Обработанное описание: '{processed_query}'")

        print(f"Исходное описание: {query_text}")
        print(f"Обработанное описание: {processed_query}")

        # Генерация эмбеддинга
        query_embedding = model.encode(
            [processed_query],
            convert_to_tensor=False,
            batch_size=1,
            show_progress_bar=False
        )[0]

        # Извлечение нарративных признаков, консистентных с базой данных
        narrative_features = extract_narrative_features_consistent(query_text, processed_query, nlp)

        return {
            "original_query": query_text,
            "processed_query": processed_query,
            "embedding": query_embedding.tolist(),
            "embedding_dimension": len(query_embedding),
            "narrative_features": narrative_features,
            "device_used": device,
            "preprocessing_applied": remove_entities
        }
    except Exception as e:
        print(f"Ошибка в основной обработке: {e}, переключаемся на fallback")
        return encode_user_query_fallback(query_text, device)


@app.function(
    image=image,
    timeout=300
)
def test_text_processing_consistency():
    """
    Тестирование консистентности обработки текста между фильмами и описанием пошьзователя
    Запуск из командной строки на локальном комп-ре:
    $ modal run modal_app.py::app.test_text_processing_consistency
    """
    import spacy
    # Импорт внутри функции для работы с добавленными директориями
    from modal_utils.cloud_operations import clean_text, prepare_text_for_embedding, encode_user_query_fallback

    nlp = spacy.load("en_core_web_lg")

    # Тестовые примеры
    test_texts = [
        "A young wizard named Harry Potter discovers his magical heritage.",
        "In New York City, a detective investigates a mysterious crime.",
        "The story follows John Smith as he travels through time.",
        "An epic adventure in the Star Wars universe with Luke Skywalker."
    ]

    print("Тестирование обработки текста:")
    print("=" * 60)

    for text in test_texts:
        processed = prepare_text_for_embedding(text, nlp)
        print(f"Исходный: {text}")
        print(f"Обработанный: {processed}")
        print("-" * 40)

    return {"test_completed": True, "samples_processed": len(test_texts)}


# Глобальная переменная для кэширования GPU индекса
_gpu_index_cache = None
_gpu_resources_cache = None


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A10G",
    timeout=300,
    min_containers=1  # Поддерживаем контейнер активным
)
def search_similar_movies(
        query_embedding: list,
        query_narrative_features: dict,
        top_k: int = 50,
        rerank_top_n: int = 10):
    """
    Поиск похожих фильмов с использованием FAISS и консистентных нарративных признаков
    дополнительным ранжированием
    по нарративным признакам. Оптимизированная версия с кэшированием GPU
    индекса для избежания повторных переносов
    """
    global _gpu_index_cache, _gpu_resources_cache
    import time
    from modal_utils.cloud_operations import (rerank_by_narrative_features,
                                              calculate_narrative_similarity)

    start_time = time.time()

    search_index = None  # Инициализируем переменную

    # Загрузка FAISS индекса
    movies_df = pd.read_parquet("/data/data/indexed_movies_metadata.parquet")

    # Инициализация GPU индекса (только при первом вызове)
    if _gpu_index_cache is None and faiss.get_num_gpus() > 0:
        print("Первая инициализация GPU индекса...")

        # Загрузка CPU индекса
        cpu_index = faiss.read_index("/data/data/movie_embeddings.index")

        load_time = time.time() - start_time
        print(f"Загрузка данных: {load_time:.3f}s")

        # Создание GPU ресурсов
        _gpu_resources_cache = faiss.StandardGpuResources()
        _gpu_resources_cache.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory

        # Перенос на GPU
        _gpu_index_cache = faiss.index_cpu_to_gpu(_gpu_resources_cache, 0, cpu_index)

        logger.info(f"GPU индекс кэширован и готов к использованию")
        print("GPU индекс кэширован и готов к использованию")
        using_gpu = True

    elif _gpu_index_cache is not None:
        logger.info(f"Используем кэшированный GPU индекс")
        print("Используем кэшированный GPU индекс")
        using_gpu = True

    else:
        logger.info(f"GPU недоступен, используем CPU")
        print("GPU недоступен, используем CPU")
        cpu_index = faiss.read_index("/data/data/movie_embeddings.index")
        search_index = cpu_index
        using_gpu = False

    if using_gpu:
        search_index = _gpu_index_cache

    # Семантический поиск, Подготовка запроса
    query_vector = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_vector)

    # Выполнение поиска ближайших соседей
    search_start = time.time()
    distances, indices = search_index.search(query_vector, top_k)
    search_time = time.time() - search_start

    logger.info(f"Время поиска ({'GPU' if using_gpu else 'CPU'}): {search_time:.3f}s")
    print(f"Время поиска ({'GPU' if using_gpu else 'CPU'}): {search_time:.3f}s")

    # Обработка результатов
    process_start = time.time()
    candidates = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(movies_df):
            movie = movies_df.iloc[idx]

            # Вычисление нарративного сходства с исправленной функцией
            narrative_similarity = calculate_narrative_similarity(
                query_narrative_features,
                movie.get('narrative_features', '{}')
            )

            candidates.append({
                'index': idx,
                'semantic_score': float(dist),
                'narrative_similarity': narrative_similarity,
                'movie_data': movie.to_dict()
            })

    # Дополнительное ранжирование с учетом нарративных признаков
    reranked_candidates = rerank_by_narrative_features(candidates)
    process_time = time.time() - process_start

    total_time = time.time() - start_time

    # Подготавливаем необходимые для инфо поля и выводим через logger
    # filtered = {}
    desired_movie_keys = {'id', 'title', 'narrative_features'}
    if reranked_candidates:  # список не пуст
        first = reranked_candidates[0]  # это dict
        movie_info = first.get("movie_data", {})  # dict с данными фильма
        filtered = {k: movie_info.get(k) for k in desired_movie_keys if k in movie_info}
        logger.info(f"First re-ranked candidate (filtered): {filtered}")

    else:
        logger.warning("reranked_candidates is empty, nothing to log")

    return {
        "results": reranked_candidates[:rerank_top_n],
        "performance_metrics": {
            "using_gpu": using_gpu,
            "search_time": search_time,
            "process_time": process_time,
            "total_time": total_time,
            "cached_gpu_index": _gpu_index_cache is not None
        }
    }


# hf_secret = modal.Secret.from_dict({"HF_MODEL_KEY": os.getenv("HF_MODEL_KEY")})

"""
@app.function(
    image=image,
    gpu="A100:2",  # 2x A100 для 70B модели
    volumes={"/model_cache": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
    min_containers=1
)
def llama_70b_instruct(
        prompt: str,
        temperature: float = 0.3,
        max_new_tokens: int = 1024
) -> str:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
    import torch

    # Конфигурация 8-битной квантизации
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,  # Двойная квантизация для лучшего сжатия
        bnb_8bit_compute_dtype=torch.float16,  # Тип данных для вычислений
        bnb_8bit_quant_type="fp8"  # Тип квантизации (доступно с bitsandbytes>=0.43.0)
    )

    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    model_path = "/model_cache/llama-3.3-70b-instruct"

    # Модель уже скачана, snapshot_download больше не нужен
    # # Загрузка модели с кэшированием
    # snapshot_download(
    #     repo_id=model_id,
    #     local_dir=model_path,
    #     ignore_patterns=["*.md", "*.txt"],
    #     token=os.getenv("HF_TOKEN")
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,    # Оптимизация CPU памяти
        trust_remote_code=True
        # load_in_4bit=True # Для 4-битной квантизации (требует bitsandbytes)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # device=0,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        return_full_text=False,
        # batch_size=1               # Уменьшение батча до 1
    )

    return pipe(prompt)[0]['generated_text']
"""

#
# @app.function(secrets=[modal.Secret.from_name("nebius-secret")])
# def run_app():
#     from app_simplified import _run_main_app
#
#     _run_main_app()
#
#     # Подготавливаем необходимые для инфо поля и выводим через logger
#     filtered = {}
#     desired_movie_keys = {'id', 'title', 'narrative_features'}
#     for candidate_key, info in reranked_candidates[0]:
#         if candidate_key == 'movie_data':
#             filtered = {k: v for k, v in info.items() if k in desired_movie_keys}
#         else:
#             filtered = info
#     logger.info(f"Первый отранжированный кандидат: {filtered}")
