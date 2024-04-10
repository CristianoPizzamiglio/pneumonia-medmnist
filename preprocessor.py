from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from keras.src.preprocessing.image import ImageDataGenerator, NumpyArrayIterator


def preprocess_data(
    pneumonia_mnist_path: Union[str, Path],
) -> Tuple[NumpyArrayIterator, NumpyArrayIterator]:
    """
    Preprocess data.

    Parameters
    ----------
    pneumonia_mnist_path : Union[str, Path]

    Returns
    -------
    Tuple[NumpyArrayIterator, NumpyArrayIterator]

    """
    raw_data = np.load(pneumonia_mnist_path)

    training_images = np.expand_dims(raw_data.f.train_images, axis=-1)
    test_images = np.expand_dims(raw_data.f.val_images, axis=-1)
    training_labels = raw_data.f.train_labels
    test_labels = raw_data.f.val_labels

    training_generator = compute_training_images_generator(
        training_images, training_labels
    )
    test_generator = compute_test_images_generator(test_images, test_labels)
    return training_generator, test_generator


def compute_training_images_generator(
    training_images: np.ndarray, training_labels: np.ndarray
) -> NumpyArrayIterator:
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


def compute_test_images_generator(
    test_images: np.ndarray, test_labels: np.ndarray
) -> NumpyArrayIterator:
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
