import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

# ========================
# Konfigurasi
# ========================
dataset_dir = "C:/S6/joji/rda/SIBI"  # Ganti path ini sesuai folder kamu
img_height, img_width = 224, 224
batch_size = 32
epochs = 50

# ========================
# Callback: EarlyStopping
# ========================
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ========================
# Preprocessing & Augmentasi
# ========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ========================
# MobileNetV2 Base Model
# ========================
base_model = MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze awal

# ========================
# Model Custom di atas MobileNetV2
# ========================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ========================
# Training awal (fit base)
# ========================
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=[early_stop]
)

# ========================
# Fine-Tuning (unfreeze bagian akhir)
# ========================
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Hanya fine-tune 30 layer terakhir
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training lanjut (fine-tuning)
history_fine = model.fit(
    train_gen,
    epochs=5,
    validation_data=val_gen,
    callbacks=[early_stop]
)

# ========================
# Simpan model
# ========================
model.save("mobilenetv2_sibi_model.h5")
print("✅ Model disimpan sebagai 'mobilenetv2_sibi_model.h5'")

# ========================
# Visualisasi & Simpan Grafik
# ========================
def plot_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")  # 💾 Simpan gambar ke file
    plt.show()
    print("📊 Grafik disimpan sebagai 'training_history.png'")

plot_history(history, history_fine)
