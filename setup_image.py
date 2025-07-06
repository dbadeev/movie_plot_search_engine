#!/usr/bin/env python3
"""
Скрипт настройки образа для PlotMatcher
"""
import os
import subprocess
import sys


def run_command(cmd):
    """Выполнение команды с проверкой ошибок"""
    print(f"Выполняется: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Ошибка: {result.stderr}")
        sys.exit(1)
    return result.stdout


def main():
    print("Настройка образа PlotMatcher...")

    # Проверка CUDA
    try:
        output = run_command("nvidia-smi")
        print("CUDA доступна:")
        print(output)
    except:
        print("Предупреждение: nvidia-smi недоступна на этапе сборки")

    # Создание необходимых директорий
    os.makedirs("/data", exist_ok=True)
    os.makedirs("/tmp/model_cache", exist_ok=True)

    print("Настройка завершена успешно!")


if __name__ == "__main__":
    main()
