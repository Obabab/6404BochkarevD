"""
Модуль интерфейса i_image_processing.py

Содержит класс IImageProcessing, определяющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Кэнни)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (метод пока не реализован)

Модуль предназначен для учебных целей (лабораторная работа по курсу "Технологии программирования на Python").
"""

from abc import ABC, abstractmethod


import numpy as np


class IImageProcessing(ABC):
    """
    Интерфейс для реализации методов обработки изображений.

    Определяет набор абстрактных методов, которые должны быть реализованы
    в наследуемых классах для выполнения различных операций над изображениями.
    """

    @abstractmethod
    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Args:
            image (np.ndarray): Входное изображение.
            kernel (np.ndarray): Ядро свёртки.

        Returns:
            np.ndarray: Результат применения свёртки к изображению.
        """
        pass

    @abstractmethod
    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Изображение в оттенках серого.
        """
        pass

    @abstractmethod
    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Значение гамма-коррекции.

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        pass

    @abstractmethod
    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            np.ndarray: Изображение с выделенными границами.
        """
        pass

    @abstractmethod
    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            np.ndarray: Изображение с выделенными углами.
        """
        pass

    @abstractmethod
    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            np.ndarray: Изображение с выделенными окружностями.
        """
        pass
