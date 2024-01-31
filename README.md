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

Image classification model for the [PneumoniaMNIST](https://zenodo.org/records/10519652) dataset.

![healthy](images/healthy.png)
![pneumonia](images/pneumonia.png)
