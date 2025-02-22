import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import random

# Параметры
IMG_SIZE = (250, 250)  # Размер изображений
BATCH_SIZE = 48
EPOCHS = 300
DATASET_PATH = "C:/Users/rusch/Py_projects/nn/ai/datas/s/lfw-deepfunneled"  # Путь к базе с папками фотографий
CSV_FILE = "C:/Users/rusch/Py_projects/nn/ai/datas/s/people.csv"            # Путь к CSV файлу с информацией

# Проверка на наличие GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Чтение CSV
df = pd.read_csv(CSV_FILE, encoding='utf-8')
print("Первые 5 строк CSV:")
print(df.head())

# Приведение столбца 'name' к строковому типу и удаление пустых значений
names = df["name"].dropna().astype(str).tolist()
print("Количество людей:", len(names))

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
    # Рандомное решение: пара из одного человека или разных
    if random.choice([True, False]):
        # Пара изображений одного человека
        person = random.choice(names)
        folder = os.path.join(DATASET_PATH, person)
        print(f"Выбрана папка для одного человека: {folder}")
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Папка не найдена: {folder}")
        images = os.listdir(folder)
        if not images:
            raise FileNotFoundError(f"В папке {folder} нет изображений.")
        # Если изображений 2 и более, выбираем случайные два; иначе используем одно изображение дважды
        if len(images) >= 2:
            image1, image2 = random.sample(images, 2)
        else:
            image1 = image2 = images[0]
        img1 = load_image(os.path.join(folder, image1))
        img2 = load_image(os.path.join(folder, image2))
        label = 1  # Изображения одного и того же человека
    else:
        # Пара изображений разных людей
        person1, person2 = random.sample(names, 2)
        folder1 = os.path.join(DATASET_PATH, person1)
        folder2 = os.path.join(DATASET_PATH, person2)
        print(f"Выбраны папки для разных людей: {folder1}, {folder2}")
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
        label = 0  # Изображения разных людей
    return img1, img2, label

# Генерация данных для обучения
def generate_training_data(batch_size):
    while True:
        images1, images2, labels = [], [], []
        for _ in range(batch_size):
            img1, img2, label = get_random_pair()
            images1.append(img1)
            images2.append(img2)
            labels.append(label)
        yield ([np.array(images1), np.array(images2)], np.array(labels))

# Создание модели
def build_model():
    base_cnn = tf.keras.applications.ResNet50(weights=None,
                                               input_shape=(250, 250, 3),
                                               include_top=False,
                                               pooling='avg')
    input1 = tf.keras.layers.Input(shape=(250, 250, 3))
    input2 = tf.keras.layers.Input(shape=(250, 250, 3))
    
    features1 = base_cnn(input1)
    features2 = base_cnn(input2)
    
    merged = tf.keras.layers.Subtract()([features1, features2])
    merged = tf.keras.layers.Dense(256, activation="relu")(merged)
    merged = tf.keras.layers.Dense(128, activation="relu")(merged)
    merged = tf.keras.layers.Dense(1, activation="sigmoid")(merged)
    
    model = tf.keras.models.Model(inputs=[input1, input2], outputs=merged)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# Обучение модели
model = build_model()
model.fit(generate_training_data(BATCH_SIZE), epochs=EPOCHS, steps_per_epoch=10)

# Сохранение модели
folder_for_save = "C:/Users/rusch/Py_projects/nn/" # сюда вставь адрес, куда сохранить
print("Назови модель для сохранения: ")
model.save(folder_for_save + input() + ".h5")