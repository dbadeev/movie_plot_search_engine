from app_simplified import _run_main_app

import modal
from modal_app import app


if __name__ == "__main__":
    """Запуск вычислений в Modal"""
    print("Запуск вычислений в Modal...")
    print("Деплоим приложение...")
    with modal.enable_output():
        app.deploy()  # ✅ Деплоим приложение
    _run_main_app()
