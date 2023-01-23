import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from functools import wraps

DATA_PATH = "data_10.json"

def log(before, after):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            print()
            print("."*10+before+"."*10)
            print()
            retval = function(*args, **kwargs)
            print()
            print("."*10 + after + "."*10)
            print()
            return retval
        return wrapper
    return decorator

@log("LOADING DATASET", "DATASET LOADED")
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    labels = np.array(data["labels"])
    mappings = np.array(data["mapping"])
    return inputs, labels, mappings


def prepare_datasets(X, y, test_size, validation_size):   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # X dimensions: (130, 13) --> convert to 3d arrays (130, 13, 1)
    X_train = X_train[..., np.newaxis] # --> (num_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_seq_model(input_shape):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])
    optimiser = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_conv_model(input_shape):
    model = keras.Sequential([

        keras.layers.Conv2D( filters=32, kernel_size=(3,3), activation="relu", input_shape=input_shape),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2, 2), padding="same"), # same is 0 padding
        keras.layers.BatchNormalization(),

        keras.layers.Conv2D( filters=32, kernel_size=(3,3), activation="relu", input_shape=input_shape),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2, 2), padding="same"), # same is 0 padding
        keras.layers.BatchNormalization(),

        keras.layers.Conv2D( filters=32, kernel_size=(2,2), activation="relu", input_shape=input_shape),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2, 2), padding="same"), # same is 0 padding
        keras.layers.BatchNormalization(),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])

    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


@log("TRAINING MODEL", "MODEL TRAINED!")
def train(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=128):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    return history


def plot_history(hist):

    fig, axs = plt.subplots(2)
    axs[0].plot(hist.history["accuracy"], label="train accuracy")
    axs[0].plot(hist.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(hist.history["loss"], label="train loss")
    axs[1].plot(hist.history["val_loss"], label="test loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss eval")

    plt.show()

def predict(model, X):
    X = X[np.newaxis, ...] # take X and make it (1, 130, 13, 1) from (130, 13, 1)
    pred = model.predict(X) # [ [0.1, 0.2, ... ] ] 10 values represents different scores for the 10 different genres
    predicted_index = np.argmax(pred, axis=1)
    return predicted_index


if __name__ == "__main__":
    X, y, mappings = load_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_datasets(X, y, test_size=0.25, validation_size=0.2)
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    print(f"Dataset length: {len(X), len(y)}")
    print(f"Inputs shape: {X.shape}")
    print(f"Unique labels: {np.unique(y)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    
    input_shape_conv = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (130, 13, 1)
    # input_shape_seq = (X_train.shape[1], X_train.shape[2]) # (130, 13)

    # model = build_seq_model(input_shape_seq)
    model = build_conv_model(input_shape_conv)
    model.summary()
    hist = train(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
    # hist = train(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32)

    test_err, test_acc = model.evaluate(X_val, y_val, verbose=1)

    print(f"\n Accuracy on test set: {test_acc}")
    # plot_history(hist)
    X_sample = X_test[100]
    y_sample = y_test[100]
    genre_i = predict(model, X_sample)
    print(f"Predicted genre: {mappings[genre_i]}")
    print(f"Expected genre: {mappings[y_sample]}")


