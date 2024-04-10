from __future__ import annotations

import datetime
import os
import pickle
import random
from dataclasses import dataclass, field

import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras.src.preprocessing.image import NumpyArrayIterator

from preprocessor import preprocess_data

random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

epoch_count: int = 40

logdir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="model.h5", monitor="val_loss", save_best_only=True
)


@dataclass
class HyperParametersSpecs:
    """
    Keras tuner hyperparameter specs.

    Parameters
    ----------
    hyper_parameters : kt.HyperParameters

    """

    hyper_parameters: kt.HyperParameters
    learning_rate: kt.HyperParameters.Choice = field(init=False)
    convolutional_layer_count: kt.HyperParameters.Int = field(init=False)
    convolutional_first_layer_filter_count: kt.HyperParameters.Int = field(init=False)
    dropout: kt.HyperParameters.Float = field(init=False)
    dense_layer_neuron_count: kt.HyperParameters.Int = field(init=False)

    def __post_init__(self) -> None:
        self.learning_rate = self.hyper_parameters.Choice(
            "learning_rate", values=[1e-4, 1e-3, 1e-2]
        )
        self.convolutional_layer_count = self.hyper_parameters.Int(
            "convolutional_layer_count", min_value=1, max_value=3, step=1
        )
        self.convolutional_first_layer_filter_count = self.hyper_parameters.Int(
            "convolutional_first_layer_filter_count",
            min_value=16,
            max_value=64,
            step=16,
        )
        self.dropout = self.hyper_parameters.Float(
            "dropout", min_value=0.1, max_value=0.5, step=0.1
        )
        self.dense_layer_neuron_count = self.hyper_parameters.Int(
            "dense_layer_neuron_count", min_value=32, max_value=512, step=32
        )


def create_parametric_model(
    hyper_parameters: kt.HyperParameters,
) -> tf.keras.models.Sequential:
    """
    Create a parametric convolutional neural network.

    Parameters
    ----------
    hyper_parameters : kt.HyperParameters

    Returns
    -------
    tf.keras.models.Sequential

    """
    hyper_parameters_specs = HyperParametersSpecs(hyper_parameters)
    model = tf.keras.models.Sequential()

    for i in range(hyper_parameters_specs.convolutional_layer_count):
        filter_count = hyper_parameters_specs.convolutional_first_layer_filter_count
        if i == 0:
            model.add(
                tf.keras.layers.Conv2D(
                    filter_count, (3, 3), activation="relu", input_shape=(28, 28, 1)
                )
            )
        else:
            model.add(
                tf.keras.layers.Conv2D(filter_count * i * 2, (3, 3), activation="relu")
            )
        model.add(tf.keras.layers.MaxPooling2D(2, 2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(hyper_parameters_specs.dropout))
    model.add(
        tf.keras.layers.Dense(
            units=hyper_parameters_specs.dense_layer_neuron_count, activation="relu"
        )
    )
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    adam_optimizer = tf.keras.optimizers.Adam(
        learning_rate=hyper_parameters_specs.learning_rate
    )
    model.compile(
        optimizer=adam_optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    return model


def tune_model(
    training_generator: NumpyArrayIterator, test_generator: NumpyArrayIterator
) -> None:
    """
    Tune the model.

    Parameters
    ----------
    training_generator : NumpyArrayIterator
    test_generator : NumpyArrayIterator

    """
    tuner = kt.Hyperband(
        create_parametric_model,
        objective="val_accuracy",
        max_epochs=epoch_count,
        project_name="pneumonia_mnist",
    )
    tuner.search_space_summary(extended=True)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    callbacks = [tensorboard_callback, model_checkpoint_callback, stop_early]

    tuner.search(
        training_generator,
        validation_data=test_generator,
        epochs=epoch_count,
        callbacks=callbacks,
    )
    tuner.results_summary()

    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    with open("best_hyperparameters.pkl", "wb") as file:
        pickle.dump(best_hyperparameters, file)


if __name__ == "__main__":
    training_generator_, test_generator_ = preprocess_data("pneumoniamnist.npz")
    tune_model(training_generator_, test_generator_)
