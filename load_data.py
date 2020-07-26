import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.callbacks import TensorBoard

print(f"Tensorflow version: {tf.__version__}\n")

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# X_train = X_train.astype('float32') / 255.0
# X_test = X_test.astype('float32') / 255.0

# y_train = y_train.astype('float32') / 255.0
# y_test = y_test.astype('float32') / 255.0

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



def build_model(hp):
    model = keras.models.Sequential()

    model.add(Conv2D(hp.Int("input_units", min_value=32, max_value=256, step=32), (3, 3), padding="same", input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int("n_layers", 1,4)):
        model.add(Conv2D(hp.Int(f"conv_{i}_units", min_value=32, max_value=256, step=32), (3, 3), padding="same"))
        model.add(Activation('relu'))

    model.add(Dropout(hp.Float('dropout', 0.0, 0.5, 0.1)))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    #model.add(Dense(hp.Int(f"dense_{i}_units", min_value=4, max_value=32, step=4)))
    model.add(Dense(hp.Choice('dense_units', values=[10, 20, 30])))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam",
               # optimizer=keras.optimizers.Adam(
               # hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    return model

import datetime
import time
import os
LOG_DIR = f"{os.getcwd()}/logs/{datetime.datetime.now().strftime('%m.%d-%H.%M')}"
tensorboard = TensorBoard(log_dir=LOG_DIR)
showtime = ShowTime()
my_callbacks = [tensorboard, showtime]

tuner = RandomSearch(
    build_model,
    objective = "val_accuracy",
    max_trials = 1,
    executions_per_trial = 1,
    directory = LOG_DIR
)

tuner.search_space_summary()

tuner.search(
    x = X_train,
    y = y_train,
    epochs = 1,
    batch_size = 64,
    callbacks=[my_callbacks],
    validation_data = (X_test, y_test)
)

tuner.results_summary()

import pickle

# Writing pickle
with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner, f)

# Reading pickle
# tuner = pickle.load(open("tuner_....pkl", "rb"))

# print(f"Tuner hyperparams:\n{tuner.get_best_hyperparameters()[0].values")
# print(tuner.results_summary())
# print(tuner.get_best_models()[0].summary())