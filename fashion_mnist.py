import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import TensorBoard

print(f"Tensorflow version: {tf.__version__}\n")

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


print("Loaded Fashion MNSIT\n")
print(f"Train X, y: {X_train.shape}, {y_train.shape}\n")
print(f"Test X, y: {X_test.shape}, {y_test.shape}\n")

class ShowTime(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        """Docstring"""
        print(f"Training started at: {datetime.datetime.now().time()}")
    def on_train_end(self, logs=None):
        print(f"Training ended at: {datetime.datetime.now().time()}")


def build_model():
    model = keras.models.Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    return model

import datetime
import os
LOG_DIR = f"{os.getcwd()}/logs/{datetime.datetime.now().strftime('%m.%d-%H.%M')}"
tensorboard = TensorBoard(log_dir=LOG_DIR)
showtime = ShowTime()
my_callbacks = [tensorboard, showtime]

model = build_model()
model.fit(X_train, y_train, epochs=20)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print("\nTest accuracy: {test_acc}")

model.save(f"fashion_mnist_model.h5")
