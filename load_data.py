import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

print(f"Tensorflow version: {tf.__version__}\n")

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print("Loaded Fashion MNSIT\n")
class ShowTime(tf.keras.callbacks.Callback):
    
    def on_train_begin(self, logs=None):
        """Docstring"""
        print(f"Training started at: {datetime.datetime.now().time()}")
    def on_train_end(self, logs=None):
        print(f"Training ended at: {datetime.datetime.now().time()}")

def build_model(hp):
    model = keras.models.Sequential()

    model.add(Conv2D(hp.Int("input_units", min_value=32, max_value=256, step=32), (3, 3), input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int("n_layers", 2,4)):
        model.add(Conv2D(hp.Int(f"conv_{i}_units", min_value=32, max_value=256, step=32), (3, 3)))
        model.add(Activation('relu'))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    return model

import time
LOG_DIR = f"{int(time.time())}"

tuner = RandomSearch(
    build_model,
    objective = "val_accuracy",
    max_trials = 1,
    executions_per_trial = 1,
    directory = LOG_DIR
)

tuner.search(
    x = X_train,
    y = y_train,
    epochs = 1,
    batch_size = 64,
    validation_data = (X_test, y_test)
)