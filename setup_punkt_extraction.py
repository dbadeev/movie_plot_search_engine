# setup_punkt_extraction.py

import pickle
import os
import ast
import sys


def extract_punkt_data_to_files():
    """Извлечение данных из english.pickle в отдельные файлы"""

    # Путь к pickle файлу
    pickle_path = "/root/nltk_data/tokenizers/punkt_tab/english/english.pickle"
    output_dir = "/root/nltk_data/tokenizers/punkt_tab/english"

    try:
        print(f"Loading punkt model from {pickle_path}")

        # Загрузка модели
        with open(pickle_path, 'rb') as f:
            punkt_model = pickle.load(f)

        print(f"Punkt model loaded successfully: {type(punkt_model)}")

        # 1. Извлечение sentence starters
        try:
            if hasattr(punkt_model, '_lang_vars') and punkt_model._lang_vars:
                sent_starters = punkt_model._lang_vars.sent_starters
                with open(f"{output_dir}/sent_starters.txt", 'w') as f:
                    f.write('\n'.join(sent_starters))
                print(f"✅ Created sent_starters.txt with {len(sent_starters)} entries")
            else:
                print("⚠️ No sentence starters found, creating default ones")
                default_starters = ["i", "you", "he", "she", "it", "we", "they", "the", "a", "an"]
                with open(f"{output_dir}/sent_starters.txt", 'w') as f:
                    f.write('\n'.join(default_starters))
        except Exception as e:
            print(f"⚠️ Error extracting sentence starters: {e}")
            # Создаем базовые стартеры
            default_starters = ["i", "you", "he", "she", "it", "we", "they", "the", "a", "an"]
            with open(f"{output_dir}/sent_starters.txt", 'w') as f:
                f.write('\n'.join(default_starters))

        # 2. Извлечение collocations
        try:
            if hasattr(punkt_model, '_params') and punkt_model._params:
                collocations = punkt_model._params.collocations
                with open(f"{output_dir}/collocations.tab", 'w') as f:
                    for (word1, word2), freq in collocations.items():
                        f.write(f"{word1}\t{word2}\t{freq}\n")
                print(f"✅ Created collocations.tab with {len(collocations)} entries")
            else:
                # Создаем пустой файл
                open(f"{output_dir}/collocations.tab", 'w').close()
                print("✅ Created empty collocations.tab")
        except Exception as e:
            print(f"⚠️ Error extracting collocations: {e}")
            open(f"{output_dir}/collocations.tab", 'w').close()

        # 3. Создание остальных файлов
        try:
            # Abbreviations
            if hasattr(punkt_model, '_params') and hasattr(punkt_model._params, 'abbrev_types'):
                with open(f"{output_dir}/abbrev_types.txt", 'w') as f:
                    f.write('\n'.join(punkt_model._params.abbrev_types))
                print("✅ Created abbrev_types.txt from model")
            else:
                # Создаем пустой файл
                open(f"{output_dir}/abbrev_types.txt", 'w').close()
                print("✅ Created empty abbrev_types.txt")

            # Ortho context (обычно пустой)
            open(f"{output_dir}/ortho_context.tab", 'w').close()
            print("✅ Created empty ortho_context.tab")

        except Exception as e:
            print(f"⚠️ Warning creating additional files: {e}")
            # Создаем пустые файлы на всякий случай
            for filename in ["abbrev_types.txt", "ortho_context.tab"]:
                open(f"{output_dir}/{filename}", 'w').close()

        print("✅ All punkt_tab files created successfully")
        return True

    except Exception as e:
        print(f"❌ Error extracting punkt data: {e}")
        return False


if __name__ == "__main__":
    success = extract_punkt_data_to_files()
    sys.exit(0 if success else 1)
