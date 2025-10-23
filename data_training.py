
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# === Настройки директорий ===
TARGET = "STOCKS"
SOURCE_DIR = os.path.join("data_yf", TARGET)
TARGET_DIR = os.path.join("data_training", TARGET)
os.makedirs(TARGET_DIR, exist_ok=True)


def process_data(df, window_size=60):
    """
    Обработка данных:
    - Приведение типов к float
    - Удаление NaN
    - Создание X_train, y_train
    - Нормализация StandardScaler на каждом окне
    """

    # Преобразуем ВСЕ столбцы в числовой формат
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna().reset_index(drop=True)

    # Берём только числовые признаки
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df = df[numeric_cols]

    if df.shape[1] == 0:
        raise ValueError("Нет числовых признаков для обучения")

    X_train, y_train = [], []

    for i in range(len(df) - window_size):
        window = df.iloc[i:i + window_size].values
        target = df.iloc[i + window_size].values

        # Нормализация по окну
        scaler = StandardScaler()
        window_scaled = scaler.fit_transform(window)

        # Нормализация таргета по тем же статистикам
        target_scaled = (target - scaler.mean_) / scaler.scale_

        X_train.append(window_scaled)
        y_train.append(target_scaled)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


# === Основной цикл ===
for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".csv"):
        filepath = os.path.join(SOURCE_DIR, filename)
        coin_name = os.path.splitext(filename)[0]

        # создаём отдельную папку
        output_dir = os.path.join(TARGET_DIR, coin_name)
        os.makedirs(output_dir, exist_ok=True)

        # лог-файл
        log_path = os.path.join(output_dir, "log.txt")
        with open(log_path, "a") as log:
            log.write(f"\n==== Обработка файла: {filename} ====\n")
            log.write(f"Время запуска: {datetime.now()}\n")

            try:
                df = pd.read_csv(filepath,skiprows=2)
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"])
                df = df.sort_values("Date").reset_index(drop=True)



                log.write(f"✅ Файл успешно прочитан. Размер: {df.shape}\n")
                log.write(f"Колонки: {list(df.columns)}\n")


                X_train, y_train = process_data(df.drop(columns=["Date"]), window_size=60)
                print(X_train, y_train)

                # === Сохраняем в .npy ===
                np.save(os.path.join(output_dir, "x_train.npy"), X_train)
                np.save(os.path.join(output_dir, "y_train.npy"), y_train)

                log.write(f"✅ Успешно сохранено:\n")
                log.write(f"   X_train: {X_train.shape}\n")
                log.write(f"   y_train: {y_train.shape}\n")
                log.write(f"   Путь: {output_dir}\n")

            except Exception as e:
                log.write(f"❌ Ошибка при обработке: {str(e)}\n")

print("\n🎉 Все файлы успешно обработаны и сохранены!")















