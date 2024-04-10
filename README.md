---
license: mit
tags:
- vision
- image-classification
- medmnist
- pneumonia
- tensorflow
- keras
datasets:
- PneumoniaMNIST
model-index:
- name: pneumonia-medmnist
  results:
  - task:
      name: Image Classification
      type: image-classification
    metrics:
      - name: Accuracy
        type: val-accuracy
        value: 0.9656488299369812
---

Convolutional neural network to classify images of the [PneumoniaMNIST](https://zenodo.org/records/10519652) dataset.
For educational purposes.

<p>
  <img src="images/healthy.png" width=300 />
  <img src="images/pneumonia.png" width=300 /> 
</p>

The model hyperparameters are tuned with [KerasTuner](https://keras.io/keras_tuner/)
using the [Hyperband](https://arxiv.org/abs/1603.06560) optimization algorithms. The 
hyperparameters are:
* Learning rate
* Number of convolutional layers
* Number of convolutional filters
* Dropout
* Number of neurons in the dense layer

A utility function is also available for plotting the feature maps (random channels):

![feature_map](images\feature_maps\3.svg)
