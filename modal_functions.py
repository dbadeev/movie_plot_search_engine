import modal
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os
import tempfile

# Конфигурация образа
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "python3-dev", "gcc", "g++", "cmake")
    .pip_install(
        "pandas==2.2.2",
        "numpy==1.26.4",
        "torch==2.2.2",
        "transformers==4.52.4",
        "sentence-transformers==2.7.0",
        "tqdm==4.66.4"
    )
    # Добавляем необходимые локальные директории напрямую в образ
    .add_local_dir("modal_utils", remote_path="/root/modal_utils")
    .add_local_dir("local_utils", remote_path="/root/local_utils")
)

app = modal.App("tmdb-processor", image=image)
volume = modal.Volume.from_name("tmdb-data")

# Пути к файлам
LOCAL_INPUT_PATH = "processed_tmdb_movies.csv"
LOCAL_OUTPUT_PATH = "processed_tmdb_movies_with_embeddings.csv"
VOLUME_INPUT_PATH = "/data/input.csv"
VOLUME_OUTPUT_PATH = "/data/output_with_embeddings.csv"


def upload_to_volume():
    """Загружает локальный файл в Modal Volume с использованием batch_upload"""
    print(f"Uploading {LOCAL_INPUT_PATH} to Modal Volume...")

    # Создаем временный файл для загрузки
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Используем batch_upload для эффективной загрузки
        with volume.batch_upload(force=True) as batch:
            # Загружаем файл в Volume
            batch.put_file(LOCAL_INPUT_PATH, VOLUME_INPUT_PATH)

        print("Upload complete")
    except Exception as e:
        print(f"Upload failed: {str(e)}")
    finally:
        # Удаляем временный файл
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def download_from_volume():
    """Скачивает файл из Modal Volume на локальную машину"""
    print(f"Downloading results to {LOCAL_OUTPUT_PATH}...")
    try:
        # Получаем список файлов для проверки
        files = volume.listdir("/data")
        print(f"Files in volume: {files}")

        # Проверяем существование файла
        if VOLUME_OUTPUT_PATH[1:] not in [f.path for f in files]:
            print(f"File {VOLUME_OUTPUT_PATH} not found in volume")
            return

        # Создаем директорию для выходного файла, если нужно
        os.makedirs(os.path.dirname(LOCAL_OUTPUT_PATH), exist_ok=True)

        # Скачиваем файл
        with open(LOCAL_OUTPUT_PATH, "wb") as f:
            for chunk in volume.read_file(VOLUME_OUTPUT_PATH):
                f.write(chunk)

        print("Download complete")
    except Exception as e:
        print(f"Download failed: {str(e)}")


@app.function(
    gpu="A10G",  # Используем GPU
    timeout=3600,  # 1 час таймаут
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("my-env")]  # для API ключей
)
def process_movies(batch_size: int = 100):
    # 1. Загрузка данных из Volume
    try:
        # Проверяем существование файла
        files = volume.listdir("/data")
        if "input.csv" not in [f.path for f in files]:
            print("Input file not found in volume")
            return {"error": "file_not_found"}

        # Читаем файл
        with open("/data/input.csv", "r") as f:
            df = pd.read_csv(f)

        print(f"Loaded {len(df)} movies")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return {"error": "data_load_failed"}

    # 2. Инициализация модели для эмбеддингов
    try:
        model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device="cuda",
            cache_folder="/root/cache"
        )

        # Переводим модель в режим половинной точности для экономии памяти
        model = model.half() if torch.cuda.is_available() else model
    except Exception as e:
        print(f"Model initialization failed: {str(e)}")
        return {"error": "model_init_failed"}

    # 3. Создание эмбеддингов для 'overview'
    try:
        embeddings = []
        non_empty_overviews = df['overview'].fillna("")

        for i in tqdm(range(0, len(non_empty_overviews), batch_size),
                      total=len(non_empty_overviews) // batch_size + 1):
            # Получаем батч текстов
            batch = non_empty_overviews.iloc[i:i + batch_size].tolist()

            # Генерируем эмбеддинги с автоматическим определением типа данных
            batch_embeddings = model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                # Убираем параметр precision, так как тип определяется автоматически
                # при использовании model.half()
            )

            # Конвертируем в float32 для совместимости с numpy
            batch_embeddings = batch_embeddings.astype(np.float32)
            embeddings.append(batch_embeddings)

    except Exception as e:
        print(f"Embedding generation failed: {str(e)}")
        return {"error": "embedding_failed"}

    # 4. Сохранение эмбеддингов
    try:
        # Объединяем все эмбеддинги
        all_embeddings = np.vstack(embeddings)

        # Сохраняем как список списков
        df['overview_embedding'] = all_embeddings.tolist()

        # Добавляем дополнительные столбцы
        df['title_length'] = df['title'].apply(len)
        df['has_overview'] = df['overview'].notna()
    except Exception as e:
        print(f"Data processing failed: {str(e)}")
        return {"error": "data_processing_failed"}

    # 5. Сохранение результата в Volume
    try:
        # Сохраняем DataFrame во временный файл
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            df.to_csv(tmp, index=False)
            tmp_path = tmp.name

            # Читаем временный файл и записываем в Volume
        with open(tmp_path, "rb") as f:
            # Используем batch_upload для записи
            with volume.batch_upload(force=True) as batch:
                batch.put_file(tmp_path, VOLUME_OUTPUT_PATH)

            # Фиксируем изменения
        volume.commit()
        print("Results saved to Volume")

    except (OSError, IOError) as e:  # Ошибки файловой системы
        print(f"File system error: {str(e)}")
        return {"error": "file_system_error"}
    except Exception as e:
        print(f"Failed to save results: {str(e)}")
        return {"error": "save_failed"}
    finally:
        # Удаляем временный файл
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {
        "processed_count": len(df),
        "embedding_dim": all_embeddings.shape[1],
        "saved_path": VOLUME_OUTPUT_PATH
    }
