import numpy as np
import tensorflow as tf
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier
from bayes_opt import BayesianOptimization
from ConvolutionalNeuralNetworkModel import train_images, train_labels, val_images, val_labels

# Configuring TensorFlow to allow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def check_and_clean_data(x, y):
    if np.isnan(x).any():
        raise ValueError("NaN values found in ", x)
    if np.isnan(y).any():
        raise ValueError("NaN values found in ", y)
    return np.nan_to_num(x), np.nan_to_num(y)


def create_model(dropout_rate=0.5, kernel_size=3, learning_rate=0.01, num_layers=2):
    model = Sequential([
        Conv2D(32, kernel_size=(int(kernel_size), int(kernel_size)), activation='relu', input_shape=(256, 256, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate)
    ])
    for _ in range(1, int(num_layers)):
        model.add(Conv2D(32, (int(kernel_size), int(kernel_size)), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(9, activation='softmax'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def fit_with(dropout_rate, learning_rate, num_layers, batch_size):
    try:
        train_images_checked, train_labels_checked = check_and_clean_data(train_images, train_labels)
        model = KerasClassifier(model=create_model, dropout_rate=dropout_rate,
                                learning_rate=learning_rate, num_layers=num_layers, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        fit_params = {
            'callbacks': [early_stopping],
            'batch_size': int(batch_size),
            'validation_data': (val_images, val_labels)
        }
        score = np.mean(cross_val_score(model, train_images_checked, train_labels_checked, cv=3, fit_params=fit_params))
        return score
    except Exception as e:
        print("An error occurred:", e)
        return -1


optimizer = BayesianOptimization(
    f=fit_with,
    pbounds={
        'dropout_rate': (0.1, 0.5),
        'learning_rate': (0.01, 0.05),
        'num_layers': (1, 3),
        'batch_size': (16, 32)
    },
    random_state=1
)
optimizer.maximize(init_points=10, n_iter=10)
