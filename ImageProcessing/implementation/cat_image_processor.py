"""
Модуль cat_image_processor.py

Содержит класс CatImageProcessor для работы с API и обработки изображений:
- запрос метаданных;
- асинхронное скачивание изображений;
- параллельная (по процессам) обработка свёрткой;
- асинхронное сохранение результатов.
"""

import os
import io
import time
import random
import functools
from typing import List, Dict, Any

import asyncio
from concurrent.futures import ProcessPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import aiohttp
import aiofiles

from PIL import Image
import numpy as np

from .cat_image import ColorCatImage, GrayscaleCatImage

# измеряет время работы
def time_logger(func): 
    @functools.wraps(func)  # декоратор
    def wrapper(*args, **kwargs):  # принимает кортеж и словарь
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        print(f"Метод {func.__name__} выполнен за {t1 - t0:.4f} секунд")
        return result
    return wrapper

# декодирует картинку и считает контуры
def process_image_worker(idx: int,
                         url: str,
                         breed_name: str,
                         image_bytes: bytes) -> Dict[str, Any]:
   
    pid = os.getpid() # получение ID текущего процесса
    print(f"Convolution for image {idx} started (PID {pid})")

    with Image.open(io.BytesIO(image_bytes)) as img: # декодировка байт в массив пикселей
        img = img.convert("RGB") # приведение к ргб
        image_array = np.array(img, dtype=np.uint8) # конвертация в nd.array(от 0 до 255)

    method = random.choice(["color", "grayscale"]) # рандомный выбор хранения и обработки
    if method == "grayscale":
        cat_img = GrayscaleCatImage(url, breed_name, image_array)
    else:
        cat_img = ColorCatImage(url, breed_name, image_array)

    # Свёртка / выделение контуров
    custom_edges = cat_img.custom_edge_detection()
    lib_edges = cat_img.library_edge_detection()

    print(f"Convolution for image {idx} finished (PID {pid})")

    return {
        "idx": idx,
        "breed_name": breed_name,
        "original": cat_img.image,
        "custom_edges": custom_edges,
        "lib_edges": lib_edges,
    }


class CatImageProcessor:
    

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._base_url = "https://api.thecatapi.com/v1/images/search"

        # HTTP-сессия с ретраями (для запроса МЕТАДАННЫХ, не картинок)
        self._session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    
    # Получение метаданных (синхронно)
    @time_logger
    def fetch_images(self, limit: int = 1) -> List[Dict[str, Any]]:
        
        headers = {"x-api-key": self.api_key}
        overfetch = max(3, limit * 4)
        params = {
            "limit": overfetch,
            "size": "small",
            "has_breeds": True,
            "mime_types": "jpg",
            "order": "RAND",
        }
        resp = self._session.get(
            self.base_url,
            headers=headers,
            params=params,
            timeout=(5, 10),
        )
        resp.raise_for_status()
        data = resp.json() or []
        data = [d for d in data if isinstance(d, dict) and d.get("url")]
        return data[:limit]


    # Старый последовательный вариант (
 
    @time_logger
    def download_image(self, url: str) -> np.ndarray:
        """
        Синхронная загрузка изображения (старый вариант).
        Сейчас не используется в асинхронном конвейере,
        оставлен для сравнения и отладки.
        """
        print(f"[sync download] GET {url}")
        resp = self._session.get(url, timeout=(5, 20))
        resp.raise_for_status()
        with Image.open(io.BytesIO(resp.content)) as img:
            img = img.convert("RGB")
            return np.array(img, dtype=np.uint8)

    @time_logger
    def process_and_save(self, images_data: List[Dict[str, Any]],
                         output_dir: str = "processed_cats_sync"):
      
        os.makedirs(output_dir, exist_ok=True)

        for idx, data in enumerate(images_data, start=1):
            url = data.get("url", "")
            breeds = data.get("breeds", []) or []
            breed_name = (
                breeds[0]["name"]
                if breeds and isinstance(breeds[0], dict) and "name" in breeds[0]
                else "Unknown"
            ).replace("/", "_")

            breed_dir = os.path.join(output_dir, breed_name)
            os.makedirs(breed_dir, exist_ok=True)

            try:
                image_array = self.download_image(url)
            except Exception as e:
                print(f"[sync skip] download failed: {url} -> {e}")
                continue

            try:
                method = random.choice(["color", "grayscale"])
                if method == "grayscale":
                    cat_img = GrayscaleCatImage(url, breed_name, image_array)
                else:
                    cat_img = ColorCatImage(url, breed_name, image_array)
            except Exception as e:
                print(f"[sync skip] prepare image failed: {url} -> {e}")
                continue

            original_path = os.path.join(
                breed_dir, f"{idx}_{breed_name}_original.png"
            )
            custom_path = os.path.join(
                breed_dir, f"{idx}_{breed_name}_custom_edges.png"
            )
            lib_path = os.path.join(
                breed_dir, f"{idx}_{breed_name}_library_edges.png"
            )

            try:
                Image.fromarray(cat_img.image).save(original_path)

                custom_edges = cat_img.custom_edge_detection()
                Image.fromarray(custom_edges).save(custom_path)

                lib_edges = cat_img.library_edge_detection()
                Image.fromarray(lib_edges).save(lib_path)
            except Exception as e:
                print(f"[sync warn] save/process failed: {url} -> {e}")
                continue

    # Асинхронно-параллельный конвейер

    # подготовка задач для обработки картинок
    def prepare_image_tasks(self,
                            images_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
      
        tasks: List[Dict[str, Any]] = []  # создание пустого списка для задач
        for idx, data in enumerate(images_data, start=1): # перебираем элементы images_data(для словаря data) и присваиваем порядк номер, начиная с 1
            url = data.get("url", "") # Берём из словаря data значение по ключу url, иначе пустая строка
            breeds = data.get("breeds", []) or [] # пытаемся взять список пород из метаданных, иначе []
            breed_name = (
                breeds[0]["name"] # порода
                if breeds and isinstance(breeds[0], dict) and "name" in breeds[0] # если список не пустой и внем ключ name, тогда взять название породы
                else "Unknown"
            ).replace("/", "_") # чтобы название породы не ломало путь к файлам и не воспринималось как часть директории

            tasks.append( # создается словарь
                {
                    "idx": idx, # номер
                    "url": url, 
                    "breed_name": breed_name, # очищенное имя породы
                }
            )
        return tasks

    async def _download_one_async(
        self,
        session: aiohttp.ClientSession, # созданная асинхронная HTTP-сессия из aiohttp
        idx: int,
        url: str,
    ) -> bytes:
    
      #  Асинхронное скачивание одного изображения
      
        print(f"Downloading image {idx} started")
        async with session.get(url, timeout=20) as resp: # гет запрос по url, максимум 20 секунд на запрос
            resp.raise_for_status() # Проверяет HTTP-статус. выбросит либо исключение, либо продолжит работу
            data = await resp.read() # асин операция. читает весь файл, возвращает байты картинки
        print(f"Downloading image {idx} finished")
        return data

    async def _save_result_async(
        self,
        output_dir: str, # куда нужно сохранить результаты обработки
        result: Dict[str, Any], # словарь с результатами(номер картинки, имя породы, исх изобр и т.д.)
    ) -> None:
        
       # Асинхронное сохранение результата (оригинал + два варианта контуров).
        
        idx = result["idx"] 
        breed_name = result["breed_name"]

        breed_dir = os.path.join(output_dir, breed_name) # Склеиваем путь к папке, где будут лежать картинки этой породы
        os.makedirs(breed_dir, exist_ok=True) # создаёт папку. True означает, что если такая папка уже есть - не считать это ошибкой

        original_path = os.path.join( # формирование пути к папкам
            breed_dir, f"{idx}_{breed_name}_original.png"
        )
        custom_path = os.path.join( # формирование пути к папкам
            breed_dir, f"{idx}_{breed_name}_custom_edges.png"
        )
        lib_path = os.path.join( # формирование пути к папкам
            breed_dir, f"{idx}_{breed_name}_library_edges.png"
        )

        async def save_array(path: str, array: np.ndarray) -> None:
            # numpy -> PNG в память
            img = Image.fromarray(array) # Превращает массив чисел array в картинку img
            buf = io.BytesIO() # файлоподобный буфер в памяти. нужен для записи картинки в png
            img.save(buf, format="PNG") # сохраняет картинку в буфер
            data = buf.getvalue() # Достаём из буфера все данные как bytes

            # асинхронная запись в файл
            async with aiofiles.open(path, "wb") as f: # асинхронно открываем файл по пути path. запись в бинарном режиме
                await f.write(data) # асинхронно записываем байты data в файл

        print(f"Saving image {idx} started")

        # составляем список тасков на запись трёх файлов
        tasks = [
            save_array(original_path, result["original"]),
            save_array(custom_path, result["custom_edges"]),
            save_array(lib_path, result["lib_edges"]),
        ]

    
        await asyncio.gather(*tasks)

        print(f"Saving image {idx} finished")


    # Полный цикл обработки 
    async def _process_single_task(
        self,
        task_info: Dict[str, Any],
        session: aiohttp.ClientSession,
        pool: ProcessPoolExecutor,
        output_dir: str,
    ) -> None:
        """
        Полный цикл для одного изображения:
        - асинхронное скачивание;
        - запуск свёртки в отдельном процессе;
        - асинхронное сохранение.
        """
        idx = task_info["idx"]
        url = task_info["url"]
        breed_name = task_info["breed_name"]

        if not url:
            print(f"[skip] image {idx}: empty url")
            return

        # 1) асинхронное скачивание
        try:
            image_bytes = await self._download_one_async(session, idx, url) # делает http запрос по url, читает ответ, возвращает байты картинки
        except Exception as e:
            print(f"[skip] image {idx}: download failed -> {e}")
            return

        # 2) CPU-часть — в процессе
        loop = asyncio.get_running_loop() # # Берём текущий event loop, который управляет всеми асинхронными задачами
        try:
            result = await loop.run_in_executor( #Запускаем (синхронную) функцию в другом потоке/процессе, ждем результат асинхронно
                pool, # рабочий процесс
                process_image_worker,
                idx,
                url,
                breed_name,
                image_bytes,
            )
        except Exception as e:
            print(f"[skip] image {idx}: processing failed -> {e}")
            return

        # 3) асинхронное сохранение
        try:
            await self._save_result_async(output_dir, result)
        except Exception as e:
            print(f"[warn] image {idx}: saving failed -> {e}")
            return

    async def _process_and_save_async_impl(
        self,
        images_data: List[Dict[str, Any]],
        output_dir: str,
    ) -> None:
        """
        Внутренняя асинхронная реализация:
        создаёт пул процессов и aiohttp-сессию,
        запускает обработку всех изображений параллельно.
        """
        tasks_info = self.prepare_image_tasks(images_data) # На выходе даёт нормализованный список
        # Дальше по tasks_info запускается асинхронная обработка всех картинок
        os.makedirs(output_dir, exist_ok=True)# Гарантируем, что корневая папка для сохранения результатов существует

        timeout = aiohttp.ClientTimeout(total=60) # для операции aiohttp таймаут 60 секунд

        # Пул процессов для свёртки
        with ProcessPoolExecutor() as pool:
            # Клиент для асинхронного HTTP
            # Создаём одну асинхронную HTTP-сессию session для всех запросов к API
            async with aiohttp.ClientSession(timeout=timeout) as session:
                coroutines = [ # создается список корутин
                    self._process_single_task(task_info, session, pool, output_dir)
                    for task_info in tasks_info # перебор списка задач. на каждой итерации - одна задача для одной картинки
                ]
                # все картинки обрабатываются "параллельно"
                """
                запускает все переданные корутины одновременно,
                даёт им выполняться параллельно (в рамках одного event loop),
                возвращает, когда все они закончатся.
                """
                await asyncio.gather(*coroutines)

    # запуск конвеера
    def process_and_save_async(
        self,
        images_data: List[Dict[str, Any]],
        output_dir: str = "processed_cats",
    ) -> None:
        """
        Публичный метод для новой версии:
        асинхронно скачивает, параллельно обрабатывает и асинхронно сохраняет
        изображения.

        Внутри сам запускает event loop через asyncio.run().
        """
        asyncio.run(self._process_and_save_async_impl(images_data, output_dir))
        """
        создаёт event loop,
        запускает в нём эту корутину,
        ждёт её завершения,
        корректно закрывает loop.
        """