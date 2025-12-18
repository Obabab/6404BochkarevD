"""
main.py

Лабораторная работа №2 по курсу "Технологии программирования на Python".

Программа скачивает изображения кошек из API, обрабатывает их выделением контуров
пользовательским и библиотечным методами, и сохраняет результаты.

Запуск:
    python main.py [limit]

Аргументы:
    limit: количество изображений для обработки (по умолчанию 1)
"""

import argparse
import os

from dotenv import load_dotenv

from implementation.cat_image_processor import CatImageProcessor

def main() -> None: # Точка входа в программу. Аннотация -> None — подсказка типов: функция ничего не возвращает.
    load_dotenv() # читает .env апи ключ
    api_key = os.getenv("API_KEY") # сохраняет значение ключа
    if not api_key:  # защита от пустого ключа
        print("Ошибка: API_KEY не найден в .env файле")
        return

    parser = argparse.ArgumentParser( # парсер документов
        description="Лабораторная работа №2: обработка изображений кошек.",
    )
    parser.add_argument(
        "limit",
        type=int,
        default=1,
        help="Количество изображений для обработки (по умолчанию 1)",
    )

    args = parser.parse_args() # Парсим фактические аргументы командной строки; доступ к ним через args.limit.

    processor = CatImageProcessor(api_key)  # Создаём процессор, передаём API-ключ.
    try:
        images_data = processor.fetch_images(limit=args.limit) # запрос к TheCatAPI,получаем список JSON-объектов
        processor.process_and_save(images_data) # 
        print("Обработка завершена.")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()