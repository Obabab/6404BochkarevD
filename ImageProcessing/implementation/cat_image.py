"""
Модуль cat_image.py

Содержит классы CatImage, ColorCatImage, GrayscaleCatImage для инкапсуляции изображений кошек.
"""

from abc import ABC, abstractmethod
import functools
import time
import numpy as np

from .image_processing import ImageProcessing
from def_implementation.def_image_processing import def_ImageProcessing


def time_logger(func): # фабрика декораторов
    @functools.wraps(func) # декоратор. для копии метаданных из функции func. С wraps они выглядят как оригинальная func
    def wrapper(*args, **kwargs): # Определяется внутренняя функция-обёртка. любые поз арг как кортеж. собирает произвольное число именованных аргументов в словарь
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Метод {func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    return wrapper 


class CatImage(ABC):
    """
    Абстрактный класс для изображений кошек.
    """

    def __init__(self, url: str, breed: str, image_array: np.ndarray):
        self._url = url
        self._breed = breed # порода
        self._custom_processor = ImageProcessing()
        self._library_processor = def_ImageProcessing()
        self._image = self.prepare_image(image_array) # prepare_image приводит картинку к нужному формату класса. цв или черно-белое

    @property # декоратор для чтения, как геттер
    def url(self) -> str:
        return self._url

    @property
    def breed(self) -> str:
        return self._breed

    @property
    def custom_processor(self) -> ImageProcessing:
        return self._custom_processor

    @property
    def library_processor(self) -> def_ImageProcessing:
        return self._library_processor

    @property
    def image(self) -> np.ndarray:
        return self._image

    @abstractmethod # абстрактный метод. полиморфизм
    def prepare_image(self, image_array: np.ndarray) -> np.ndarray:
        pass

    @time_logger # обертка
    def custom_edge_detection(self) -> np.ndarray:
        return self.custom_processor.edge_detection(self.image)

    @time_logger
    def library_edge_detection(self) -> np.ndarray:
        return self.library_processor.edge_detection(self.image)

    
    def __add__(self, other) -> 'CatImage':
        if isinstance(other, CatImage):
            new_image = np.clip(self.image.astype(np.int32) +
                            other.image.astype(np.int32), 0, 255).astype(np.uint8)
            return self.__class__(self.url, f"{self.breed}+{other.breed}", new_image)
        elif isinstance(other, (int, float)):
            new_image = np.clip(self.image.astype(np.int32) + other, 0, 255).astype(np.uint8)
            return self.__class__(self.url, f"{self.breed}+{other}", new_image)
        raise TypeError(f"Unsupported type: {type(other)}")

    def __sub__(self, other: 'CatImage') -> 'CatImage':
        if isinstance(other, CatImage): # проверка правого операнда принадлежит ли он CatImage
            new_image = np.clip(self.image.astype(np.int32) - # переводим каналы из uint8 в знаковый тип большего разряда, чтобы не было переполнения по модулю 256
                                other.image.astype(np.int32), 0, 255).astype(np.uint8)
            return self.__class__(self.url, f"{self.breed}-{other.breed}", new_image)
        raise TypeError

    def __str__(self) -> str:
        return f"CatImage: breed={self.breed}, url={self.url}, shape={self.image.shape}"


class ColorCatImage(CatImage):
    """Класс для цветных изображений кошек."""
    def prepare_image(self, image_array: np.ndarray) -> np.ndarray:
        if image_array.ndim == 3: # проверка на ргб
            return image_array # если трехмерный, то просто возвращаем
        return np.stack([image_array, image_array, image_array], axis=-1) # добавление новой оси, если черно-белое. 
        # собирает несколько массивов одинаковой формы в новое измерение

class GrayscaleCatImage(CatImage):
    """Класс для ч/б изображений кошек."""
    def prepare_image(self, image_array: np.ndarray) -> np.ndarray:
        if image_array.ndim == 3: # проверка черно-белая ли картинка
            self._image = self.custom_processor._rgb_to_grayscale(image_array).astype(np.uint8) # перевод в черно-белое
            return self.image
        return image_array


class Dog:
    def sound(self):
        return "Гав"

class Cat:
    def sound(self):
        return "Мяу"

def make_sound(animal):  
    print(animal.sound())

make_sound(Dog())  
make_sound(Cat())  # Мяу