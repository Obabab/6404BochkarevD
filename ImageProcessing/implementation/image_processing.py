"""
Модуль image_processing.py

Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

Содержит класс ImageProcessing, предоставляющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Кэнни)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (метод пока не реализован)

Модуль предназначен для учебных целей (лабораторная работа по курсу "Технологии программирования на Python").
"""

import cv2

import interfaces

import numpy as np


class ImageProcessing(interfaces.IImageProcessing):
    """
    Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

    Предоставляет методы для обработки изображений, включая свёртку, преобразование
    в оттенки серого, гамма-коррекцию, а также обнаружение границ, углов и окружностей.

    Методы:
        _convolution(image, kernel): Выполняет свёртку изображения с ядром.
        _rgb_to_grayscale(image): Преобразует RGB-изображение в оттенки серого.
        _gamma_correction(image, gamma): Применяет гамма-коррекцию.
        edge_detection(image): Обнаруживает границы (Canny).
        corner_detection(image): Обнаруживает углы (Harris).
        circle_detection(image): Обнаруживает окружности (HoughCircles).
    """

    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Использует функцию cv2.filter2D для применения ядра к изображению.

        Args:
            image (np.ndarray): Входное изображение (может быть цветным или чёрно-белым).
            kernel (np.ndarray): Ядро свёртки (матрица).

        Returns:
            np.ndarray: Изображение после применения свёртки.
        """
        return cv2.filter2D(image, -1, kernel)

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Использует функцию cv2.cvtColor для преобразования цветного изображения
        в чёрно-белое.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Коррекция осуществляется с помощью таблицы преобразования значений пикселей.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гамма-коррекции (>0).

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255
                          for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Использует оператор Кэнни (cv2.Canny) для выделения границ.
        Предварительно изображение преобразуется в оттенки серого.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """
        gray = self._rgb_to_grayscale(image)
        edges = cv2.Canny(gray, 100, 200)
        return edges

    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении.

        Использует алгоритм Харриса (cv2.cornerHarris) для поиска углов.
        Углы выделяются красным цветом на копии исходного изображения.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными углами (красные точки).
        """
        gray = self._rgb_to_grayscale(image)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        result = image.copy()
        result[dst > 0.01 * dst.max()] = [255, 0, 0]
        return result

    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении.

        Использует преобразование Хафа (cv2.HoughCircles) для поиска окружностей.
        Найденные окружности выделяются зелёным цветом, центры — красным.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Изображение с выделенными окружностями.
        """
        raise NotImplementedError("Метод обнаружения окружностей пока не реализован.")
