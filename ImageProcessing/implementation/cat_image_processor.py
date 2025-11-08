"""
Модуль cat_image_processor.py

Содержит класс CatImageProcessor для работы с API и обработки изображений.
"""

import os
import io
import time
import random
import functools
from typing import List, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
import numpy as np

from .cat_image import ColorCatImage, GrayscaleCatImage


def time_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        print(f"Метод {func.__name__} выполнен за {t1 - t0:.4f} секунд")
        return result
    return wrapper


class CatImageProcessor:
    """
    Класс для обработки изображений кошек: запрос метаданных в API, скачивание,
    обработка (контуры) и сохранение результатов.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._base_url = "https://api.thecatapi.com/v1/images/search"

        # HTTP-сессия с ретраями
        self._session = requests.Session() #Создаем "умное" HTTP-соединение
        retries = Retry(
            total=3,  # МАКСИМУМ 3 попытки
            backoff_factor=0.5, # Пауза между попытками
            status_forcelist=[429, 500, 502, 503, 504],  # При каких ошибках повторять
            allowed_methods=["GET"], # Только для GET-запросов
            raise_on_status=False,   # Не выбрасывать исключение после всех попыток
        )
        adapter = HTTPAdapter(max_retries=retries) #создание адаптера. как прослойка между программой и интернетом
        self._session.mount("http://", adapter) # адаптер для HTTP
        self._session.mount("https://", adapter) # адаптер для HTTPS

    @property # декоратор. для превращения в атрибут объекта
    def api_key(self) -> str:
        return self._api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    @time_logger # декоратор
    def fetch_images(self, limit: int = 1) -> List[Dict[str, Any]]: # хотим получить не более limit элементов (по умолчанию 1).
        """
        Возвращает JSON-метаданные изображений котов.
        Забираем с запасом (overfetch) и только лёгкие форматы.
        """
        headers = {"x-api-key": self.api_key}
        overfetch = max(3, limit * 4)
        params = {
            "limit": overfetch,
            "size": "small",
            "has_breeds": True,
            "mime_types": "jpg",
            "order": "RAND",
        }
        resp = self._session.get(self.base_url, headers=headers, params=params, timeout=(5, 10))
        resp.raise_for_status()
        data = resp.json() or []
        data = [d for d in data if isinstance(d, dict) and d.get("url")]
        return data[:limit]

    @time_logger
    def download_image(self, url: str) -> np.ndarray:
        """
        Скачивает изображение по прямому URL с таймаутами, конвертирует в RGB.
        """
        print(f"[download] GET {url}")
        resp = self._session.get(url, timeout=(5, 20))
        resp.raise_for_status()
        with Image.open(io.BytesIO(resp.content)) as img:
            img = img.convert("RGB")
            return np.array(img, dtype=np.uint8)

    @time_logger
    def process_and_save(self, images_data: List[Dict[str, Any]], output_dir: str = "processed_cats"):
        """
        Скачивает, обрабатывает и сохраняет изображения:
        - original.png
        - *_custom_edges.png  (самописный Sobel)
        - *_library_edges.png (OpenCV Canny)
        """
        os.makedirs(output_dir, exist_ok=True)

        target = len(images_data)   # сколько котов хотим сохранить
        saved = 0                   # сколько реально сохранили в ЭТОМ запуске

        for data in images_data:
            url = data.get("url", "")
            breeds = data.get("breeds", []) or []
            breed_name = (breeds[0]["name"] if breeds and isinstance(breeds[0], dict) and "name" in breeds[0]
                        else "Unknown").replace("/", "_")

            breed_dir = os.path.join(output_dir, breed_name)
            os.makedirs(breed_dir, exist_ok=True)

            # 1) скачать
            try:
                image_array = self.download_image(url)
            except Exception as e:
                print(f"[skip] download failed: {url} -> {e}")  # если не получилось скачать - скип
                continue

            # 2) выбрать контейнер
            try:
                method = random.choice(["color", "grayscale"])
                cat_img = GrayscaleCatImage(url, breed_name, image_array) if method == "grayscale" else ColorCatImage(url, breed_name, image_array)
            except Exception as e:
                print(f"[skip] prepare image failed: {url} -> {e}")
                continue

            # 3) имена формируем по количеству УСПЕШНО сохранённых
            idx = saved + 1
            try:
                original_path = os.path.join(breed_dir, f"{idx}_{breed_name}_original.png") # сохранение оригинала
                Image.fromarray(cat_img.image).save(original_path)

                print("[process] custom_edge_detection...")
                custom_edges = cat_img.custom_edge_detection() # вызов обработки каст методом
                custom_path = os.path.join(breed_dir, f"{idx}_{breed_name}_custom_edges.png")
                Image.fromarray(custom_edges).save(custom_path)

                print("[process] library_edge_detection...")
                lib_edges = cat_img.library_edge_detection() # вызов обработки библ методом
                lib_path = os.path.join(breed_dir, f"{idx}_{breed_name}_library_edges.png")
                Image.fromarray(lib_edges).save(lib_path)

                saved += 1
                if saved >= target: # Проверяет, достигли ли мы нужного количества
                    break
            except Exception as e: # ловит ошибку и продолжает работу
                print(f"[warn] save/process failed: {url} -> {e}")
                # не увеличиваем saved — идём дальше
                continue

