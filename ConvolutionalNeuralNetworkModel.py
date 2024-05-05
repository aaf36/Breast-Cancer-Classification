from data_processing import train_images, train_labels, val_images, val_labels, test_images, test_labels
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np

train_images = train_images.reshape(-1, 256, 256, 1)
val_images = val_images.reshape(-1, 256, 256, 1)

train_labels = train_labels - 1
val_labels = val_labels - 1

train_labels = to_categorical(train_labels, num_classes=9)
val_labels = to_categorical(val_labels, num_classes=9)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(9, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(train_images, train_labels, epochs=20, batch_size=64, validation_data=(val_images, val_labels))

