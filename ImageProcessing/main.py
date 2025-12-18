import argparse
import os
import time

from dotenv import load_dotenv
from implementation.cat_image_processor import CatImageProcessor
from logging_config import setup_logging


def main() -> None:
    load_dotenv()
    api_key = os.getenv("API_KEY")

    # Парсер аргументов командной строки
    parser = argparse.ArgumentParser(
        description="обработка изображений кошек (sync vs async+multiprocessing).",
    )
    parser.add_argument(
        "limit",
        type=int,
        help="Количество изображений для обработки",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Сравнить время последовательной и асинхронно-параллельной обработки",
    )
    parser.add_argument(
        "--log-config",
        default="logging.yml",
        help="Путь к YAML-файлу с конфигурацией логгера (по умолчанию logging.yml)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Имя файла для логов (переопределяет значение из logging.yml)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Директория для файла логов (переопределяет значение из logging.yml)",
    )

    args = parser.parse_args()

    # Настраиваем логгер через YAML-конфиг + опциональные переопределения
    logger = setup_logging(
        config_path=args.log_config,
        log_file=args.log_file,
        log_dir=args.log_dir,
    )

    if not api_key:
        logger.error("Ошибка: API_KEY не найден в .env файле")
        return

    processor = CatImageProcessor(api_key)

    try:
        logger.info("Старт программы: получение метаданных изображений")
        # 1. Получаем метаданные (общие для обеих версий)
        images_data = processor.fetch_images(limit=args.limit)

        if args.compare:
            # --- Последовательная версия ---
            logger.info("=== Последовательная обработка (sync) ===")
            t2 = time.perf_counter()
            processor.process_and_save(
                images_data,
                output_dir="processed_cats_sync",
            )
            t3 = time.perf_counter()
            logger.info(
                "Последовательная обработка завершена за %.2f секунд.",
                t3 - t2,
            )

        # --- Async + multiprocessing версия ---
        logger.info("=== Async + multiprocessing обработка ===")
        t0 = time.perf_counter()
        processor.process_and_save_async(
            images_data,
            output_dir="processed_cats_async",
        )
        t1 = time.perf_counter()
        logger.info(
            "Обработка (async + multiprocessing) завершена за %.2f секунд.",
            t1 - t0,
        )
        logger.info("Программа успешно завершена.")

    except Exception as e:
        # logger.exception сам добавит traceback
        logger.exception("Необработанное исключение: %s", e)


if __name__ == "__main__":
    main()
