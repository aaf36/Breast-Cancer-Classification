import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix


# Define paths to your training, validation, and test data directories
train_dir = 'train'
validation_dir = 'val'
test_dir = 'test'

# Create an instance of ImageDataGenerator for data augmentation (for training data)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create an instance of ImageDataGenerator for validation and test data (no data augmentation, just rescaling)
test_val_datagen = ImageDataGenerator(rescale=1./255)

# Setup the generators to read images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # match the image input size of MobileNetV2
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_val_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Load MobileNetV2 from Keras applications
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # A dense layer with 1024 units
x = Dropout(0.5)(x)  # Dropout layer for regularization
predictions = Dense(9, activation='softmax')(x)  # Final layer with softmax activation for 9 classes

# Define the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#############################################################################################################
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Callback to save the best model in Keras format
checkpoint = ModelCheckpoint(
    'CNN_transfer_model1.keras',  # Change the extension from .h5 to .keras
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

# Callback to stop training early if no improvement in validation loss
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    mode='min'
)

callbacks_list = [checkpoint, early_stopping]

##############################################################################
# Start training the model
history = model.fit(
    train_generator,
    epochs=50,  # You can adjust this based on how the model performs
    validation_data=validation_generator,
    callbacks=callbacks_list,
    verbose=1
)

