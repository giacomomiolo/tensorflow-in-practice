import tensorflow as tf
print(tf.__version__)

class ShowTime(Callback):
    
    def on_train_begin(self, logs=None):
        """Docstring"""
        print(f"Training started at: {datetime.datetime.now().time()}")
    def on_train_end(self, logs=None):
        print(f"Training ended at: {datetime.datetime.now().time()}")

print("Hello")
