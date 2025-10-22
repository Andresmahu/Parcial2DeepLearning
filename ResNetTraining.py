import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras import layers, models, optimizers
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Guardar el mejor modelo según val_accuracy
checkpoint_cb = ModelCheckpoint(
    "Models/mejor_modelo.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Early stopping basado en val_accuracy
earlystop_cb = EarlyStopping(
    monitor='val_accuracy',  # detiene si no mejora la accuracy
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Reduce LR si la val_accuracy se estanca
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint_cb, earlystop_cb, reduce_lr_cb]

# CONFIGURACIÓN INICIAL
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

df = pd.read_csv('data/train/etiquetas_train.csv')
valid_df = pd.read_csv('data/valid/etiquetas_valid.csv')
train_dir = "data/train/train_cropped"
val_dir = "data/valid/valid_cropped"

# DATA AUGMENTATION + NORMALIZACIÓN
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,          # normaliza los pixeles
    rotation_range=15,        # rotaciones aleatorias
    width_shift_range=0.1,    # desplazamiento horizontal
    height_shift_range=0.1,   # desplazamiento vertical
    zoom_range=0.1,           # zoom aleatorio
    horizontal_flip=True,     # voltea horizontalmente
    fill_mode='nearest'       # relleno
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

#GENERADORES (con resize automático)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=train_dir,
    x_col='filename',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory=val_dir,
    x_col='filename',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=True,
    seed=42
)

# MODELO ResNet101 (Transfer Learning)
base_model = ResNet101(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Congelar las capas convolucionales
base_model.trainable = False

#CLASIFICADOR PERSONALIZADO
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])


# COMPILACIÓN (usa GPU automáticamente si está disponible)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ENTRENAMIENTO

history = model.fit(
    train_generator,
    validation_data=val_generator,
    callbacks=callbacks,
    epochs=20
)

model.save("resnet_transfer_learning.h5")
