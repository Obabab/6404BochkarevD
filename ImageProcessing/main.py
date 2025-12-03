import argparse
import os
import time

from dotenv import load_dotenv
from implementation.cat_image_processor import CatImageProcessor


def main() -> None:
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Ошибка: API_KEY не найден в .env файле")
        return

    parser = argparse.ArgumentParser(
        description="Лабораторная №2: обработка изображений кошек (sync vs async+multiprocessing).",
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

    args = parser.parse_args()

    processor = CatImageProcessor(api_key)

    try:
        # 1. Получаем метаданные (общие для обеих версий)
        images_data = processor.fetch_images(limit=args.limit)

        if args.compare:
            # --- СТАРАЯ ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ ---
            print("\n=== Последовательная обработка (sync) ===")
            t2 = time.perf_counter()
            processor.process_and_save(
                images_data,
                output_dir="processed_cats_sync",
            )
            t3 = time.perf_counter()
            print(
                f"Последовательная обработка завершена за {t3 - t2:.2f} секунд.\n"
            )

        # --- НОВАЯ ASYNC + MULTIPROCESSING ВЕРСИЯ ---
        print("=== Async + multiprocessing обработка ===")
        t0 = time.perf_counter()
        processor.process_and_save_async(
            images_data,
            output_dir="processed_cats_async",
        )
        t1 = time.perf_counter()
        print(
            f"Обработка (async + multiprocessing) завершена за {t1 - t0:.2f} секунд."
        )

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
