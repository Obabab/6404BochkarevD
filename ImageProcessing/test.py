from PIL import Image
import numpy as np
from implementation.cat_image import ColorCatImage

p1 = r"D:\6404_BochkarevD\ImageProcessing\processed_cats\Bombay\2_Bombay_original.png"
p2 = r"D:\6404_BochkarevD\ImageProcessing\processed_cats\Havana Brown\1_Havana Brown_original.png"

# Загружаем изображения и приводим ко всем одному размеру
img1 = Image.open(p1).convert("RGB")
img2 = Image.open(p2).convert("RGB").resize(img1.size, Image.BILINEAR)

# Создаём объекты
a = ColorCatImage(p1, "BreedA", np.array(img1, np.uint8))
b = ColorCatImage(p2, "BreedB", np.array(img2, np.uint8))

# Печатаем информацию о каждом (вызовет __str__)
print(a)
print(b)

# Операции
s = a + 100
d = a - b

# Печатаем информацию о результатах
print(s)
print(d)

# Сохраняем файлы
Image.fromarray(s.image).save("sum.png")
Image.fromarray(d.image).save("diff.png")

print("OK: sum.png и diff.png сохранены")
