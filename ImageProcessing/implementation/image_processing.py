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

from interfaces import IImageProcessing

import numpy as np


class ImageProcessing(IImageProcessing):
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
        k_h, k_w = kernel.shape   # размер ядра в виде кортежа
        pad_h, pad_w = k_h // 2, k_w // 2 # ширина и высота рамки паддинга

        # Добавляем padding
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

        # Создаем выходной массив
        output = np.zeros_like(image)

        # Применяем свертку. Суть: каждый пиксель изображения заменяется взвешенной суммой его соседей
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i + k_h, j:j + k_w] # срез матрицы с рамкой
                output[i, j] = np.sum(region * kernel) # происходит произведение среза изображения и заданного ядра

        return output

    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray: # Вычисление яркости пикселя и перевод в серый
        """
        Преобразует RGB-изображение в оттенки серого.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.
        """
        #broadcasting
        return np.clip(0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2], 0, 255).astype(
            np.float32) # взвешанная сумма  RGB каналов. массив приводится к типу float32

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
        # Проверка входных параметров. значение гаммы было положительным (нельзя делить на 0 или возводить в отрицательную степень)
        if gamma <= 0:
            raise ValueError("Gamma must be greater than 0")

        # Нормализуем изображение до диапазона [0, 1]. ТЧоб не было переполнения
        normalized = image.astype(np.float32) / 255.0 # Большинство изображений в памяти — это 8-битные значения 0…255



        # Применяем гамма-коррекцию
        # Если gamma > 1 - изображение становится темнее
        # Если gamma < 1 - изображение становится светлее
        corrected = np.power(normalized, 1.0 / gamma) # каждый пиксель массива возводится в степень 1.0 / gamma

        # Возвращаем к диапазону [0, 255] и преобразуем обратно в uint8
        # Если этого не сделать изображение будет непонятно интерпретироваться, не отобразится
        result = (corrected * 255).astype(np.uint8)

        return result

    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Использует оператор Собеля для выделения границ.
        Предварительно изображение преобразуется в оттенки серого.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Двоичное (чёрно-белое) изображение с выделенными границами.
        """

        # 1. Преобразование в оттенки серого
        gray = self._rgb_to_grayscale(image)

        # 2. Определяем операторы Собеля
        sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
        sobel_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32)

        # 3. Считаем градиенты по X и Y
        grad_x = self._convolution(gray, sobel_x)
        grad_y = self._convolution(gray, sobel_y)

        # 4. Модуль градиента (величина изменения яркости)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 5. Нормализация в диапазон [0, 255]
        if gradient_magnitude.max() > 0:
            gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255

        # 6. Бинаризация — превращаем в строго чёрно-белое изображение
        # Порог 50 можно изменить под контрастность картинки
        binary_edges = (gradient_magnitude > 50).astype(np.uint8) * 255

        # 7. Возвращаем результат
        return binary_edges


    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении с помощью детектора Харриса.
        """
        k = 0.04 # Параметр Харриса

        # 1. Конвертируем в оттенки серого 
        gray = self._rgb_to_grayscale(image)  # Переход к одному каналу (яркости), т.к. Харрис работает на интенсивности.

        # 2. Вычисляем производные
        # позволяют найти, как сильно изменяется яркость:
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) 
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) 

        Ix = self._convolution(gray, sobel_x) # изменение яркости по х
        Iy = self._convolution(gray, sobel_y) # изменение яркости по y

        # 3. Вычисляем элементы матрицы структуры
        # узнать интенсивность изменений в окрестности точки — то есть, не в одной точке, а в маленьком "окне" вокруг неё.
        Ix2 = Ix * Ix 
        Iy2 = Iy * Iy 
        Ixy = Ix * Iy # показывает, изменяется ли яркость одновременно по X и Y

        # 4. Гауссово сглаживание. 
        gaussian_kernel = np.array([[1, 2, 1],  # Это Гауссов фильтр 3×3 — приближение нормального распределения.
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=np.float32) / 16.0 # Чтобы при свёртке не изменялась общая яркость, ядро нужно нормировать, чтобы в итоге было 1
        # Чтобы детектор Харриса не срабатывал на шум, а реагировал на структуру, мы сглаживаем значения в окрестности каждого пикселя
        Sx2 = self._convolution(Ix2, gaussian_kernel)
        Sy2 = self._convolution(Iy2, gaussian_kernel)
        Sxy = self._convolution(Ixy, gaussian_kernel)

        # 5. Вычисляем отклик Харриса
        det = Sx2 * Sy2 - Sxy * Sxy
        trace = Sx2 + Sy2  
        R = det - k * trace * trace # формула отклика Харриса. показывает, насколько пиксель похож на угол

        # 6. Нормализация
        R_norm = (R - np.min(R)) / (np.max(R) - np.min(R))  # Линейная нормализация R в диапазон [0,1] для удобства порогования/сравнения в пределах кадра.

        # 7. Адаптивный подбор порога
        def find_adaptive_threshold(R_norm, target_corners=1000): # порог 10000
            """Автоматически подбирает порог для получения нужного количества углов"""
            # Сортируем значения отклика по убыванию
            sorted_response = np.sort(R_norm.flatten())[::-1] # превращает матрицу в один длинный список всех чисел. сортирует эти значения по убыванию

            # Берем порог, соответствующий target_corners-ному пикселю
            if len(sorted_response) > target_corners:
                threshold = sorted_response[target_corners] # берет значение порога на 1000 позиции в сортированном списке
            else:
                threshold = sorted_response[-1] if len(sorted_response) > 0 else 0.5  # записывается наименьшая точка отсортированного списка если длина этого списка больше 0 иначе в переменную запишется 0.5

            return max(0.1, min(0.9, threshold))  # Ограничиваем диапазон. чтоб порог был от 0.1 до 0.9


        # Автоматически подбираем порог для ~1000 углов
        threshold = find_adaptive_threshold(R_norm, target_corners=1000)
        print(f"Adaptive threshold: {threshold:.3f}") # число с 3 знаками после запятой
        corner_mask = R_norm > threshold  # матрица, хрнаящая значение true/false. является  ли пиксель углом

        print(f"R range: {np.min(R):.6f} - {np.max(R):.6f}") # выводится диапазон длины от мин до макс с 6 числами после точки
        print(f"R_norm range: {np.min(R_norm):.6f} - {np.max(R_norm):.6f}")
        print(f"Pixels above threshold: {np.sum(corner_mask)}") # выводит кол-во пикселей выше порога


        # 8. Подавление немаксимумов
        height, width = R.shape
        local_maxima = np.zeros_like(corner_mask, dtype=bool) # пустая матрица для записи углов

        for i in range(1, height - 1):  #проход по всем пикселям, кроме рамки по краям
            for j in range(1, width - 1):
                if corner_mask[i, j]:  # Проверяем: этот пиксель вообще считался углом после порога
                    neighborhood = R_norm[i - 1:i + 2, j - 1:j + 2] # Берём маленькое окно 3×3 вокруг текущего пикселя.
                    if R_norm[i, j] == np.max(neighborhood): # является ли значение текущего пикселя максимальным среди соседей
                        local_maxima[i, j] = True

        print(f"Local maxima found: {np.sum(local_maxima)}")

        # 9. Создаем результат - ВАЖНО: создаем копию в правильном формате, чтобы программа могла рисовать углы поверх копии
        result = image.copy().astype(np.uint8) # от 0 до 255

        # 10. Получаем координаты углов
        corner_coords = np.where(local_maxima)

        # 11. Рисуем углы - УВЕЛИЧИВАЕМ РАЗМЕР ТОЧЕК
        if len(corner_coords[0]) > 0:  # проверяет, есть ли вообще найденные углы
            for y, x in zip(corner_coords[0], corner_coords[1]): # объединяет два массива для обхода каждой точки по y,x
                # Рисуем квадрат 3x3 вокруг пикселя  
                # Используются max() и min(), чтобы не выйти за границы изображения
                y_start = max(0, y - 1)
                y_end = min(height, y + 2)
                x_start = max(0, x - 1)
                x_end = min(width, x + 2)

                # Значение 255 в красном канале и 0 в остальных делает точку ярко-красной, но рисует BGR, поэтому угол будет синий
                result[y_start:y_end, x_start:x_end, 0] = 255  # Красный
                result[y_start:y_end, x_start:x_end, 1] = 0  # Зеленый
                result[y_start:y_end, x_start:x_end, 2] = 0  # Синий

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
