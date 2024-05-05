from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.applications import MobileNetV2
# Define paths to your directories
train_dir = 'train'
val_dir = 'val'

# Define ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

# Load images from directories with the correct target size
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Function to create and modify the model, include any parameters you want to tune
def create_model(learning_rate=0.001, dropout_rate=0.0, neurons=512):
    # Load MobileNetV2 without the top layer
    base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False  # Freeze the base model to not train it

    # Rebuild the top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # This layer now correctly connects to the last layer of MobileNetV2
    x = Dense(neurons, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(9, activation='softmax')(x)  # Assuming 9 classes

    # Compile the new model
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

learning_rates = [0.0005, 0.001, 0.0015]   # Adding 0.0015 for finer control
batch_sizes = [64, 128]  # Focusing on larger batch sizes for efficiency
dropout_rates = [0.0, 0.2]  # Simplified to two options
neurons_options = [256, 384]  # Concentrating on potentially optimal neuron counts
layer_options = [1, 3]

best_score = 0
best_params = {}

# Iterating over all combinations of parameters
for lr in learning_rates:
    for dr in dropout_rates:
        for neurons in neurons_options:
            print(f"Testing model with learning rate: {lr}, dropout rate: {dr}, neurons: {neurons}")
            model = create_model(learning_rate=lr, dropout_rate=dr, neurons=neurons)
            history = model.fit(train_generator, validation_data=val_generator, epochs=10)

            # Assuming validation accuracy is what we're looking to maximize
            validation_accuracy = max(history.history['val_accuracy'])
            print("Validation accuracy: ", validation_accuracy)

            if validation_accuracy > best_score:
                best_score = validation_accuracy
                best_params = {'learning_rate': lr, 'dropout_rate': dr, 'neurons': neurons}

print("Best parameters: ", best_params)
print("Best validation accuracy: ", best_score)
