from __future__ import annotations

import pickle
import random

import numpy as np
import tensorflow as tf
from keras.src.preprocessing.image import NumpyArrayIterator

from hyperparameter_tuning import (
    create_parametric_model,
    model_checkpoint_callback,
    tensorboard_callback,
    epoch_count,
)
from plotter import plot_image
from preprocessor import preprocess_data

random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


def fit_model(
    training_generator: NumpyArrayIterator,
    test_generator: NumpyArrayIterator,
    model: tf.keras.models.Sequential,
) -> None:
    """
    Fit model.

    Parameters
    ----------
    training_generator : NumpyArrayIterator
    test_generator : NumpyArrayIterator
    model : tf.keras.models.Sequential

    """
    callbacks = [tensorboard_callback, model_checkpoint_callback]
    history = model.fit(
        training_generator,
        validation_data=test_generator,
        epochs=epoch_count,
        callbacks=callbacks,
    )
    best_epoch_index = history.history["val_loss"].index(
        min(history.history["val_loss"])
    )
    print(
        f"Epoch Index: {best_epoch_index}\n"
        f"Validation Loss: {history.history['val_loss'][best_epoch_index]}\n"
        f"Validation Accuracy: {history.history['val_accuracy'][best_epoch_index]}"
    )

    with open("history.pkl", "wb") as file:
        pickle.dump(history.history, file)


if __name__ == "__main__":
    training_generator_, test_generator_ = preprocess_data("pneumoniamnist.npz")

    images, labels = next(training_generator_)
    index = random.randint(0, len(images))
    plot_image(images[index], labels[index])

    with open("best_hyperparameters.pkl", "rb") as file:
        best_hyperparameters = pickle.load(file)

    model_ = create_parametric_model(best_hyperparameters)
    model_.summary()

    fit_model(training_generator_, test_generator_, model_)
