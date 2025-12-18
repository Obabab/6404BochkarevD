import unittest
import numpy as np

from implementation.cat_image import ColorCatImage, GrayscaleCatImage


class TestCatImage(unittest.TestCase):
    """Тесты для классов-наследников CatImage."""

    # ===== Вспомогательные методы =====

    def _create_grayscale_img(self):
        """Создаём 2x2 RGB и GrayscaleCatImage на её основе."""
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)  # все пиксели чёрные (0)
        rgb[0, 0] = [10, 20, 30]  # меняем один пиксель в позиции [0,0]

        img = GrayscaleCatImage(
            url="http://example.com/cat.jpg",
            breed="TestBreed",
            image_array=rgb,
        )
        return img, rgb

    def _create_color_img(self):
        """Создаём случайную RGB-картинку и ColorCatImage."""
        rgb = np.random.randint(0, 256, size=(4, 5, 3), dtype=np.uint8)
        img = ColorCatImage(
            url="http://example.com/cat.jpg",
            breed="ColorBreed",
            image_array=rgb,
        )
        return img, rgb

    def _add_two_color_images(self):
        """Складываем две ColorCatImage с разными значениями пикселей."""
        a = np.full((2, 2, 3), 200, dtype=np.uint8)
        b = np.full((2, 2, 3), 100, dtype=np.uint8)

        img1 = ColorCatImage("url1", "Breed1", a)
        img2 = ColorCatImage("url2", "Breed2", b)

        result = img1 + img2
        return result

    def _create_subtraction_results(self):
        """Готовим результаты вычитания для тестов разности картинок."""
        # первая картинка ярче
        a = np.full((2, 2, 3), 150, dtype=np.uint8)
        # вторая темнее
        b = np.full((2, 2, 3), 100, dtype=np.uint8)

        img1 = ColorCatImage("url1", "Breed1", a)
        img2 = ColorCatImage("url2", "Breed2", b)

        result1 = img1 - img2  # 150 - 100 = 50

        # для проверки клиппинга
        c = np.full((2, 2, 3), 50, dtype=np.uint8)
        img3 = ColorCatImage("url3", "Breed3", c)
        result2 = img3 - img2  # 50 - 100 -> 0 с обрезкой

        return result1, result2

    def _get_str_representation(self):
        """Создаем ColorCatImage и возвращаем строку и параметры."""
        arr = np.zeros((3, 4, 3), dtype=np.uint8)  # чёрная картинка
        url = "http://example.com/cat.png"
        breed = "PrettyCat"

        img = ColorCatImage(url, breed, arr)
        s = str(img)
        return s, url, breed

    # ===== Тесты GrayscaleCatImage =====

    def test_grayscale_image_is_2d(self): # стала ли картинка двумерной
        img, _ = self._create_grayscale_img()
        self.assertEqual(img.image.ndim, 2)

    # ===== Тесты сложения =====

    def test_add_is_color_images(self): # будет ли картинка ColorCatimage
        result = self._add_two_color_images()
        self.assertIsInstance(result, ColorCatImage)

    def test_add_breed(self): # конкатенация породы
        result = self._add_two_color_images()
        self.assertEqual(result.breed, "Breed1+Breed2")

    def test_add_clipped_to_255(self): # пиксели должны быть 255 после clip
        result = self._add_two_color_images()
        self.assertTrue(np.all(result.image == 255))

    # ===== Тесты вычитания =====

    def test_sub_is_color_images(self):
        result1, _ = self._create_subtraction_results() # будет ли картинка ColorCatImage
        self.assertIsInstance(result1, ColorCatImage)

    def test_sub_breed(self):
        result1, _ = self._create_subtraction_results()# конкатенация породы
        self.assertEqual(result1.breed, "Breed1-Breed2")

    def test_sub_pixels_50(self): # будут ли пиксели = 50
        result1, _ = self._create_subtraction_results()
        self.assertTrue(np.all(result1.image == 50))

    def test_sub_clipping_zero(self): # будут ли пиксели = 0 после clip
        _, result2 = self._create_subtraction_results()
        self.assertTrue(np.all(result2.image == 0))

    # ===== Тесты строкового представления =====

    def test_str_prefix_catimage(self): # содержит ли строка CatImage
        s, url, breed = self._get_str_representation()
        self.assertIn("CatImage:", s)

    def test_str_breed(self): # содержит ли породу
        s, url, breed = self._get_str_representation()
        self.assertIn(f"breed={breed}", s)

    def test_str_url(self): # содержит ли ссылку
        s, url, breed = self._get_str_representation()
        self.assertIn(f"url={url}", s)

    def test_str_shape(self): # содержит ли размер массива картинки
        s, url, breed = self._get_str_representation()
        self.assertIn("shape=(3, 4, 3)", s)


if __name__ == "__main__":
    unittest.main()
