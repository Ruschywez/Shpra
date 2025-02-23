from PIL import Image
import torch
from facenet_pytorch import MTCNN

# Инициализируем детектор
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=False, device=device)

def crop_face(image_path):
    # Открываем изображение
    img = Image.open(image_path).convert("RGB")

    # Получаем координаты лица
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        x1, y1, x2, y2 = map(int, boxes[0])  # Берём первое лицо
        img = img.crop((x1, y1, x2, y2))  # Обрезаем по границам
    else:
        print("Лицо не найдено, возвращаю оригинал.")

    # Ресайзим до 160x160
    img = img.resize((160, 160))

    return img

# Исправленный путь
img = crop_face(r"C:\Users\rusch\Downloads\images.jpg")  # Путь исправлен
img.show()  # Показывает изображение
