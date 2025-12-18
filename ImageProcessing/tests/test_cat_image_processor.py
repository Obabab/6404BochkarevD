import unittest
from unittest import mock
import tempfile
import os
import io

import numpy as np
from PIL import Image

from implementation.cat_image_processor import CatImageProcessor


class TestCatImageProcessorFetch(unittest.TestCase):
    """Тесты синхронной части CatImageProcessor (fetch_images)."""

    def _prepare_processor_and_result(self, limit=2):
        """
        Общая подготовка:
        - создаём CatImageProcessor с фейковым API-ключом;
        - подменяем _session.get;
        - вызываем fetch_images и возвращаем (processor, result).
        """
        processor = CatImageProcessor(api_key="DUMMY") # создаем процессор с ненастоящим ключом

        fake_data = [ # что будет возвращать поддельный response.json
            {"url": "http://example.com/1.jpg"},
            {"url": "http://example.com/2.jpg"},
            {"something": "no_url"},
            "garbage",
        ]

        class FakeResponse: # фейковый ответ
            def raise_for_status(self):
                pass

            def json(self):
                return fake_data 

        # подмена сессии
        processor._session = mock.MagicMock()
        processor._session.get.return_value = FakeResponse()

        result = processor.fetch_images(limit=limit)
        return processor, result

    def test_fetch_images_calls_get(self):
        processor, result = self._prepare_processor_and_result(limit=2)
        self.assertTrue(processor._session.get.called)

    def test_fetch_images_limit(self): # сделать срез и вернуть 2 элемента
        processor, result = self._prepare_processor_and_result(limit=2)
        self.assertEqual(len(result), 2) 

    def test_fetch_images_returns_dicts(self): # возвращает словари
        processor, result = self._prepare_processor_and_result(limit=2)
        self.assertTrue(all(isinstance(item, dict) for item in result))

    def test_fetch_images_have_url(self): # каждый элемент результата обязан содержать url
        processor, result = self._prepare_processor_and_result(limit=2)
        self.assertTrue(all("url" in item for item in result))


class TestCatImageProcessorAsync(unittest.IsolatedAsyncioTestCase):
    """Тесты асинхронных методов CatImageProcessor."""

    async def _run_save_and_collect(self):
        """
        Общая логика для _save_result_async:
        - создаём маленькую картинку;
        - вызываем _save_result_async во временную папку;
        - возвращаем (dir_exists, files).
        """
        processor = CatImageProcessor(api_key="DUMMY")

        base = np.full((2, 2, 3), 128, dtype=np.uint8)

        result = {
            "idx": 1,
            "breed_name": "TestBreed",
            "original": base,
            "custom_edges": base,
            "lib_edges": base,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            await processor._save_result_async(tmpdir, result)

            breed_dir = os.path.join(tmpdir, "TestBreed")
            dir_exists = os.path.isdir(breed_dir)
            files = os.listdir(breed_dir) if dir_exists else []
            return dir_exists, files

    async def test_creates_breed_directory(self): # обязан создать папку с породой
        dir_exists, files = await self._run_save_and_collect()
        self.assertTrue(dir_exists)

    async def test_creates_three_files(self): # создает originan, custom_edges, lib_edges
        dir_exists, files = await self._run_save_and_collect()
        self.assertEqual(len(files), 3)

    async def test_name_file_with_prefixes(self): # имена файлов начинаются с ожидаемых префиксов
        dir_exists, files = await self._run_save_and_collect()

        expected_prefixes = { # набор нужных префиксов
            "1_TestBreed_original",
            "1_TestBreed_custom_edges",
            "1_TestBreed_library_edges",
        }
        found_prefixes = { # оставляем только те, по которым нашёлся хоть один файл
            prefix
            for prefix in expected_prefixes
            if any(f.startswith(prefix) for f in files)
        }
        self.assertEqual(found_prefixes, expected_prefixes)


# =========================
# Доп. задание 2
# Интеграционный тест с Mock-CatImage
# =========================

class FakeCatImage: # Это поддельный класс, который выглядит как CatImage
    """
    Упрощённый Mock-объект CatImage для интеграционного теста.
    Нужен, чтобы интеграционный тест проверял логику пайплайна
    Не использует настоящие процессоры, просто:
    - хранит image;
    - методы custom_edge_detection / library_edge_detection возвращают тот же массив.
    """

    def __init__(self, url: str, breed: str, image_array: np.ndarray):
        self.url = url
        self.breed = breed
        self.image = image_array

    def custom_edge_detection(self) -> np.ndarray:
        return self.image

    def library_edge_detection(self) -> np.ndarray:
        return self.image


class TestCatImageProcessorIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Интеграционный тест CatImageProcessor:
    нужен для того, чтобы прогнать весь асинхронный конвейер
    - используем Mock-реализацию CatImage (FakeCatImage) вместо настоящих Color/Grayscale;
    - подменяем загрузку изображения, чтобы не ходить в реальный интернет;
    - запускаем _process_and_save_async_impl и проверяем, что файлы созданы.
    """

    async def _run_full_async_pipeline(self):
        processor = CatImageProcessor(api_key="DUMMY")

        # Подготовим картинку и закодируем в PNG-байты
        arr = np.full((4, 4, 3), 200, dtype=np.uint8) # создание картинки
        buf = io.BytesIO() # сохраняем в PNG в буфер Bytes
        Image.fromarray(arr).save(buf, format="PNG")
        png_bytes = buf.getvalue() # имитация, что скачали картинку по сети

        async def fake_download_one_async(self, session, idx, url): # асинхронная заглушка, которая игнорирует session/idx/url
            return png_bytes # возвращает байты

        #подменяем методы и классы
        with mock.patch.object(CatImageProcessor, "_download_one_async", fake_download_one_async), \
             mock.patch("implementation.cat_image_processor.ColorCatImage", FakeCatImage), \
             mock.patch("implementation.cat_image_processor.GrayscaleCatImage", FakeCatImage):

            images_data = [  # имитация того, что вернул бы fetch_images
                {
                    "url": "http://example.com/fake1.jpg", # одна картинка
                    "breeds": [{"name": "IntegrationBreed"}], # одна порода
                }
            ]
            # во временную директорию запускаем реальный _process_and_save_async_impl
            with tempfile.TemporaryDirectory() as tmpdir:
                await processor._process_and_save_async_impl(images_data, tmpdir)

                breed_dir = os.path.join(tmpdir, "IntegrationBreed")
                dir_exists = os.path.isdir(breed_dir)
                files = os.listdir(breed_dir) if dir_exists else []
                return dir_exists, files

    async def test_async_creates_breed_directory(self): # должна появиться папка IntegrationBreed. но после теста она пропадет
        dir_exists, files = await self._run_full_async_pipeline()
        self.assertTrue(dir_exists)

    async def test_async_creates_three_files(self): # должно появиться 3 файла(original + 2 варианта контуров)
        dir_exists, files = await self._run_full_async_pipeline()
        self.assertEqual(len(files), 3)


if __name__ == "__main__":
    unittest.main()
