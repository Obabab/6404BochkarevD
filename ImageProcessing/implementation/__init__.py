"""
Пакет implementation: высокоуровневые классы для работы с изображениями кошек.
"""

from .cat_image_processor import CatImageProcessor
from .cat_image import CatImage, ColorCatImage, GrayscaleCatImage

__all__ = ["CatImageProcessor", "CatImage", "ColorCatImage", "GrayscaleCatImage"]
