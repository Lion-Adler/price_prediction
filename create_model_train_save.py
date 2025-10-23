
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from datetime import datetime
import pandas as pd

# === Параметры ===
BASE_DIR = "data/data_training/CRYPTO"
SAVE_DIR = "modelsave"
EPOCHS = 50
BATCH_SIZE = 64
MIN_SAMPLES = 100  # минимум обучающих примеров


def build_model():
    """Создание архитектуры LSTM + Transformer + MLP"""
    inp = layers.Input(shape=(60, 5))

    # === LSTM блок ===
    x = layers.LSTM(60, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)(inp)
    x = layers.LSTM(60, activation='tanh', return_sequences=False)(x)
    x = layers.Reshape((60, 1))(x)
    x = layers.Dropout(0.1)(x)
    lstm_out = x

    # === Transformer Encoder ===
    norm1 = layers.LayerNormalization(epsilon=1e-6)(lstm_out)
    attn_out = layers.MultiHeadAttention(num_heads=5, key_dim=120, dropout=0.15)(norm1, norm1)
    attn_out = layers.Dropout(0.15)(attn_out)
    res1 = layers.Add()([attn_out, lstm_out])

    norm2 = layers.LayerNormalization(epsilon=1e-6)(res1)
    dense1 = layers.Dense(5, activation='relu')(norm2)
    dense1 = layers.Dropout(0.15)(dense1)
    dense2 = layers.Dense(60, activation='linear')(dense1)
    res2 = layers.Add()([dense2, res1])

    # === MLP Decoder ===
    x = layers.GlobalAveragePooling1D(data_format='channels_first')(res2)
    x = layers.Dropout(0.10)(x)
    x = layers.Dense(30, activation='relu')(x)
    out = layers.Dense(5, activation='linear')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model



def train_and_save_model(coin_dir, coin_name):
    """Обучает и сохраняет модель (с явной валидацией)"""
    x_path = os.path.join(coin_dir, "x_train.npy")
    y_path = os.path.join(coin_dir, "y_train.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"⚠️ Пропуск {coin_name}: отсутствуют x_train.npy или y_train.npy")
        return

    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)

    if len(X) < MIN_SAMPLES:
        print(f"⚠️ Пропуск {coin_name}: слишком мало данных ({len(X)} < {MIN_SAMPLES})")
        log_path = os.path.join(SAVE_DIR, coin_name, "log_warning.txt")
        os.makedirs(os.path.join(SAVE_DIR, coin_name), exist_ok=True)
        with open(log_path, "w") as log:
            log.write(f"⚠️ Недостаточно данных для обучения {coin_name}\n")
            log.write(f"Размер X: {X.shape}, y: {y.shape}\n")
            log.write(f"Требуется минимум: {MIN_SAMPLES}\n")
        return

    # === Деление на train/val ===
    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    print(f"\n🧠 Обучение модели для {coin_name}")
    print(f"   X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"   X_val:   {X_val.shape}   | y_val:   {y_val.shape}")

    model = build_model()

    # === Папка для сохранения ===
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(SAVE_DIR, coin_name)
    os.makedirs(save_path, exist_ok=True)

    # === Коллбэки ===
    log_dir = os.path.join(save_path, f"logs_{time_tag}.txt")
    checkpoint_path = os.path.join(save_path, f"model_{time_tag}.keras")

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(save_path, f"history_{time_tag}.csv"))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1
    )

    # === Обучение ===
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[csv_logger, checkpoint]
    )

    # === Сохраняем финальную модель ===
    final_model_path = os.path.join(save_path, f"final_model_{time_tag}.keras")
    model.save(final_model_path)

    # === Лог ===
    with open(log_dir, "w") as log:
        log.write(f"🧠 Обучение модели для {coin_name}\n")
        log.write(f"Время запуска: {datetime.now()}\n")
        log.write(f"Размеры данных:\n")
        log.write(f"  X_train: {X_train.shape}, y_train: {y_train.shape}\n")
        log.write(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}\n")
        log.write(f"Количество эпох: {EPOCHS}\n")
        log.write(f"Финальная модель: {final_model_path}\n")
        log.write(f"Лучший чекпоинт: {checkpoint_path}\n")
        log.write("\n📊 История обучения:\n")
        df = pd.DataFrame(history.history)
        log.write(df.tail().to_string())

    print(f"✅ Модель {coin_name} обучена и сохранена в {save_path}")

# === Основной цикл ===
if __name__ == "__main__":
    for coin_name in os.listdir(BASE_DIR):
        coin_dir = os.path.join(BASE_DIR, coin_name)
        if os.path.isdir(coin_dir):
            train_and_save_model(coin_dir, coin_name)

    print("\n🎯 Все модели обучены и сохранены!")

