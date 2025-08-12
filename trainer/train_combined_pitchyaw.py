#!/usr/bin/env python
import os
import csv
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import RandomTranslation
from keras.optimizers import Adam

from settings import MAX_OFFSET, BATCH_SIZE, IMG_SIZE, INPUT_SHAPE, DATASET_DIR, OUTPUT_DIR, LEARNING_RATE, EPOCHS

GAZE_DATASET_DIR = os.path.join(DATASET_DIR, "gaze")

def load_labels(csv_path):
    rows = []
    with open(csv_path, "r", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    return rows

class EyeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, rows, dataset_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=True):
        self.rows = rows
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.rows))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.rows) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        for i in batch_indices:
            row = self.rows[i]
            left_img_path = os.path.join(self.dataset_dir, row["left_image"])
            right_img_path = os.path.join(self.dataset_dir, row["right_image"])
            theta1 = float(row["theta1"])
            theta2 = float(row["theta2"])

            left_img = Image.open(left_img_path).convert("L").resize(self.img_size)
            right_img = Image.open(right_img_path).convert("L").resize(self.img_size)
            left_img = np.array(left_img) / 255.0
            right_img = np.array(right_img) / 255.0
            left_img = np.expand_dims(left_img, axis=-1)
            right_img = np.expand_dims(right_img, axis=-1)
            combined_img = np.concatenate([left_img, right_img], axis=1)
            batch_images.append(combined_img)
            batch_labels.append([theta1, theta2])
        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def main():
    # Load all label rows
    csv_path = os.path.join(GAZE_DATASET_DIR, "labels.csv")
    all_rows = load_labels(csv_path)

    # Split into train/test
    indices = np.arange(len(all_rows))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_rows = [all_rows[i] for i in train_idx]
    test_rows = [all_rows[i] for i in test_idx]

    # Further split train into train/val
    val_split = int(0.1 * len(train_rows))
    val_rows = train_rows[:val_split]
    train_rows = train_rows[val_split:]

    train_gen = EyeDataGenerator(train_rows, GAZE_DATASET_DIR, batch_size=BATCH_SIZE, shuffle=True)
    val_gen = EyeDataGenerator(val_rows, GAZE_DATASET_DIR, batch_size=BATCH_SIZE, shuffle=False)
    test_gen = EyeDataGenerator(test_rows, GAZE_DATASET_DIR, batch_size=BATCH_SIZE, shuffle=False)

    # Build the model
    model = Sequential([
        InputLayer(input_shape=INPUT_SHAPE),
        RandomTranslation(height_factor=MAX_OFFSET/128, width_factor=MAX_OFFSET/256, fill_mode="constant"),
        Conv2D(32, (7, 7), activation='relu'),
        MaxPooling2D((3, 3)),
        Conv2D(64, (7, 7), activation='relu'),
        MaxPooling2D((3, 3)),
        Conv2D(128, (7, 7), activation='relu'),
        MaxPooling2D((3, 3)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, name='gaze-c')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    model.summary()

    # Callbacks
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=6, verbose=1, min_lr=1e-6
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[lr_scheduler, early_stopping]
    )

    results = model.evaluate(test_gen)
    print("Test loss and MAE:", results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_save_path = os.path.join(OUTPUT_DIR, "combined_pitchyaw.h5")
    model.save(model_save_path)
    print("Model saved to", model_save_path)

if __name__ == '__main__':
    main()
