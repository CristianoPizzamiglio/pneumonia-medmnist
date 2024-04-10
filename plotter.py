from __future__ import annotations

import random

import numpy as np
import tensorflow as tf
from keras.src.preprocessing.image import NumpyArrayIterator
from matplotlib import pyplot as plt

from preprocessor import preprocess_data


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


def plot_cnn_features(
    model: tf.keras.models.Sequential, image_generator: NumpyArrayIterator
) -> None:
    """
    Plot features learnt during training.

    Parameters
    ----------
    model : tf.keras.models.Sequential
    image_generator : NumpyArrayIterator

    """
    images, labels = next(image_generator)
    index = random.randint(0, len(images) - 1)
    image = images[index]
    image = image.reshape((1,) + image.shape)

    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(image)

    first_conv_layer_activation = activations[0]
    first_pooling_layer_activation = activations[1]
    first_channel_index = random.randint(0, first_conv_layer_activation.shape[3] - 1)

    second_conv_layer_activation = activations[2]
    second_pooling_layer_activation = activations[3]
    second_channel_index = random.randint(0, second_conv_layer_activation.shape[3] - 1)

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].matshow(first_conv_layer_activation[0, :, :, first_channel_index])
    axes[0, 0].set_title("First Convolutional Layer", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].matshow(first_pooling_layer_activation[0, :, :, first_channel_index])
    axes[0, 1].set_title("First Pooling Layer", fontsize=10)
    axes[0, 1].axis("off")

    axes[1, 0].matshow(second_conv_layer_activation[0, :, :, second_channel_index])
    axes[1, 0].set_title("Second Convolutional Layer", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].matshow(second_pooling_layer_activation[0, :, :, second_channel_index])
    axes[1, 1].set_title("Second Pooling Layer", fontsize=10)
    axes[1, 1].axis("off")

    fig.suptitle("Feature Maps")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    training_generator_, test_generator_ = preprocess_data("pneumoniamnist.npz")
    model_ = tf.keras.models.load_model("model.h5")
    plot_cnn_features(model_, training_generator_)
