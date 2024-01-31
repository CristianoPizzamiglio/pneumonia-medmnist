from __future__ import annotations

import datetime
import os
import pickle
import random

import numpy as np
import tensorflow as tf
from keras.src.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from matplotlib import pyplot as plt

random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


def plot_image(image: np.ndarray, one_hot_label: int) -> None:
    """
    Plot an image.

    Parameters
    ----------
    image : np.ndarray
    one_hot_label : int

    """
    plt.imshow(image, cmap="gray")
    label = "Healthy" if one_hot_label == 0 else "Pneumonia"
    plt.title(label)
    plt.axis("off")
    plt.show()


def compute_training_images_generator(training_images: np.ndarray, training_labels: np.ndarray) -> NumpyArrayIterator:
    """
    Compute training image generator.

    Parameters
    ----------
    training_images : np.ndarray
    training_labels : np.ndarray

    Returns
    -------
    NumpyArrayIterator

    """
    training_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    return training_datagen.flow(training_images, training_labels)


def compute_test_images_generator(test_images: np.ndarray, test_labels: np.ndarray) -> NumpyArrayIterator:
    """
    Compute test image generator.

    Parameters
    ----------
    test_images : np.ndarray
    test_labels : np.ndarray

    Returns
    -------
    NumpyArrayIterator

    """
    training_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return training_datagen.flow(test_images, test_labels)


def create_model() -> tf.keras.models.Sequential:
    """
    Create convolutional neural network.

    Returns
    -------
    tf.keras.models.Sequential

    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def fit_model(
    training_generator: NumpyArrayIterator, test_generator: NumpyArrayIterator, model: tf.keras.models.Sequential
) -> None:
    """
    Fit model.

    Parameters
    ----------
    training_generator : NumpyArrayIterator
    test_generator : NumpyArrayIterator
    model : tf.keras.models.Sequential

    """
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model.h5", monitor="val_loss", save_best_only=True
    )
    callbacks = [tensorboard_callback, model_checkpoint_callback]
    history = model.fit(training_generator, validation_data=test_generator, epochs=40, callbacks=callbacks)

    with open("history.pkl", "wb") as file:
        pickle.dump(history.history, file)


if __name__ == "__main__":
    raw_data = np.load("pneumoniamnist.npz")

    training_images_ = np.expand_dims(raw_data.f.train_images, axis=-1)
    test_images_ = np.expand_dims(raw_data.f.val_images, axis=-1)
    training_labels_ = raw_data.f.train_labels
    test_labels_ = raw_data.f.val_labels

    index = random.randint(0, len(training_images_))
    plot_image(training_images_[index], training_labels_[index])

    training_generator_ = compute_training_images_generator(training_images_, training_labels_)
    test_generator_ = compute_test_images_generator(test_images_, test_labels_)

    model_ = create_model()
    fit_model(training_generator_, test_generator_, model_)
