import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import random
import matplotlib.pyplot as plt

# Параметры
IMG_SIZE = (250, 250)  # Размер изображений
DATASET_PATH = "C:/Users/rusch/Py_projects/nn/ai/datas/s/lfw-deepfunneled"  # Путь к базе с папками фотографий
CSV_FILE = "C:/Users/rusch/Py_projects/nn/ai/datas/s/people.csv"            # Путь к CSV файлу с информацией
MODEL_FILE = "C:/Users/rusch/Py_projects/nn/piska.h5"  # Файл сохранённой модели

# Вывод текущей рабочей директории (чтобы знать, куда сохраняется модель и график)
print("Текущая рабочая директория:", os.getcwd())

# Получение настроек дообучения от пользователя
user_batch = input("Введите размер батча для дообучения (например, 32): ")
user_epochs = input("Введите количество эпох для дообучения (например, 10): ")
user_save_interval = input("Введите интервал сохранения (количество шагов, например, 100): ")

BATCH_SIZE = int(user_batch)
EPOCHS = int(user_epochs)
SAVE_INTERVAL = int(user_save_interval)

# Проверка на наличие GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Чтение CSV и подготовка списка имен
df = pd.read_csv(CSV_FILE, encoding='utf-8')
names = df["name"].dropna().astype(str).tolist()
print("Количество людей:", len(names))
print("Первые 5 строк CSV:")
print(df.head())

# Функция для загрузки изображения
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # Нормализация
    return img

# Функция для создания случайной пары изображений
def get_random_pair():
    # С вероятностью 50% выбираем пару изображений одного человека
    if random.choice([True, False]):
        person = random.choice(names)
        folder = os.path.join(DATASET_PATH, person)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Папка не найдена: {folder}")
        images = os.listdir(folder)
        if not images:
            raise FileNotFoundError(f"В папке {folder} нет изображений.")
        # Если изображений 2 и более, выбираем два разных, иначе используем одно изображение дважды
        if len(images) >= 2:
            image1, image2 = random.sample(images, 2)
        else:
            image1 = image2 = images[0]
        img1 = load_image(os.path.join(folder, image1))
        img2 = load_image(os.path.join(folder, image2))
        label = 1  # Один и тот же человек
    else:
        # Пара изображений разных людей
        person1, person2 = random.sample(names, 2)
        folder1 = os.path.join(DATASET_PATH, person1)
        folder2 = os.path.join(DATASET_PATH, person2)
        if not os.path.exists(folder1):
            raise FileNotFoundError(f"Папка не найдена: {folder1}")
        if not os.path.exists(folder2):
            raise FileNotFoundError(f"Папка не найдена: {folder2}")
        images1 = os.listdir(folder1)
        images2 = os.listdir(folder2)
        if not images1:
            raise FileNotFoundError(f"В папке {folder1} нет изображений.")
        if not images2:
            raise FileNotFoundError(f"В папке {folder2} нет изображений.")
        image1 = random.choice(images1)
        image2 = random.choice(images2)
        img1 = load_image(os.path.join(folder1, image1))
        img2 = load_image(os.path.join(folder2, image2))
        label = 0  # Разные люди
    return img1, img2, label

# Генератор данных для обучения
def generate_training_data(batch_size):
    while True:
        images1, images2, labels = [], [], []
        for _ in range(batch_size):
            img1, img2, label = get_random_pair()
            images1.append(img1)
            images2.append(img2)
            labels.append(label)
        yield ([np.array(images1), np.array(images2)], np.array(labels))

# Загрузка ранее сохранённой модели
print("Загрузка модели из файла:", MODEL_FILE)
model = tf.keras.models.load_model(MODEL_FILE)

# Вывод информации о модели и количестве параметров
model.summary()

# Создание callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_FILE,     # Модель будет сохраняться в этот файл
    save_freq=SAVE_INTERVAL, # Сохраняем каждые SAVE_INTERVAL шагов
    verbose=1
)

csv_logger = tf.keras.callbacks.CSVLogger('training_log.csv', append=True)

callbacks_list = [checkpoint_callback, csv_logger]

# Дообучение модели с перехватом KeyboardInterrupt
try:
    history = model.fit(
        generate_training_data(BATCH_SIZE),
        epochs=EPOCHS,
        steps_per_epoch=10,  # Можно настроить и это значение
        callbacks=callbacks_list
    )
except KeyboardInterrupt:
    print("Обучение прервано пользователем.")
finally:
    # Финальное сохранение модели (на случай, если обучение завершилось между сохранениями)
    model.save(MODEL_FILE)
    print("Обновлённая модель сохранена в файле:", MODEL_FILE)
    
    # Построение графика динамики точности
    try:
        log_df = pd.read_csv('training_log.csv')
        if 'accuracy' in log_df.columns:
            plt.figure(figsize=(8, 6))
            plt.plot(log_df['epoch'], log_df['accuracy'], marker='o', label='Точность на обучении')
            plt.xlabel('Эпоха')
            plt.ylabel('Точность')
            plt.title('Динамика точности по эпохам')
            plt.legend()
            plt.grid(True)
            plt.savefig('accuracy_plot.png')
            plt.show()
            print("График сохранён в файле: accuracy_plot.png")
        else:
            print("В CSV файле не найден столбец 'accuracy'.")
    except Exception as e:
        print("Не удалось построить график:", e)
