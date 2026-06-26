# Landslide Detection using U-Net

This project implements a U-Net convolutional neural network for landslide detection from satellite imagery. It utilizes multi-spectral images (RGB, NDVI) along with geographical features like slope and elevation to segment landslide areas.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Setup](#setup)
3.  [Data Loading and Preprocessing](#data-loading-and-preprocessing)
4.  [Model Architecture](#model-architecture)
5.  [Training](#training)
6.  [Prediction and Evaluation](#prediction-and-evaluation)

## Introduction

Landslides are natural hazards that cause significant damage and loss of life. Accurate and timely detection of landslides is crucial for disaster management and risk assessment. This project uses a deep learning approach, specifically a U-Net model, to identify landslide-affected regions in satellite images. The model takes into account various spectral bands and topographic information to improve detection accuracy.

## Setup

To run this notebook, ensure you have the necessary libraries installed. The following commands install `tensorflow` and `matplotlib`.

```python
!pip install tensorflow
!pip install matplotlib
```

You will also need to mount your Google Drive to access the dataset. The notebook assumes the dataset is located at `/content/gdrive/MyDrive/DL/landslide4Sense`.

## Data Loading and Preprocessing

The dataset consists of HDF5 files containing satellite images and corresponding mask files for landslide areas. The images include RGB, NIR, slope, and elevation data. During preprocessing, the following steps are performed:

- NaN values are replaced with a small float number.
- RGB, slope, and elevation data are normalized.
- NDVI (Normalized Difference Vegetation Index) is calculated using the Red and NIR bands.
- The input features are combined into a 6-channel array (Red, Green, Blue, NDVI, Slope, Elevation).
- The data is split into training and validation sets.

## Model Architecture

The U-Net model is employed for semantic segmentation. It consists of an encoder (contraction path) to capture context and a decoder (expansive path) to enable precise localization. Dropout layers are included to prevent overfitting. The model is compiled with the Adam optimizer and a custom `dice_loss` function, along with `accuracy`, `precision`, `recall`, and `f1_m` metrics.

## Training

The model is trained for 100 epochs with a batch size of 16. A `ModelCheckpoint` callback is used to save the best model weights based on the `val_f1_m` metric.

## Prediction and Evaluation

After training, the model is used to make predictions on the validation set. The predictions are binarized using a threshold of 0.5. The performance of the model is evaluated using loss, accuracy, precision, recall, and F1-score on the validation data. Visualizations of the training history (loss, precision, recall, F1-score over epochs) are also provided.
