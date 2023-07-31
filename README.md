# Classification, Localization, Segmentation for PASCAL VOC 2012

This repository contains code for building and training a neural network for classification, localization, and segmentation tasks using the PASCAL VOC 2012 dataset. The goals of this project are to showcase the flexibility of neural networks with multiple heterogeneous outputs, including bounding boxes, class labels, and masks for segmentation.

## Dataset

The dataset used for this project is the PASCAL VOC 2012 dataset, which can be obtained from [Kaggle](https://www.kaggle.com/huanghanchina/pascal-voc-2012).

## Model Building Steps

### Step 1: Extract Label Annotations

In the first step, we extract label annotations from the PASCAL VOC 2012 dataset. These annotations include class labels and bounding box coordinates for the detected objects in the images.

### Step 2: Precompute Convolutional Representations

Next, we use a pre-trained image classification model, specifically ResNet50, to precompute convolutional representations for all the images in the object detection training set. The representations will have a shape of (7, 7, 2048) for each image.

### Step 3: Baseline Object Detection Model

In the final step, we design and train a baseline object detection model with two heads to predict:

1. Class labels of the detected objects.
2. Bounding box coordinates of a single detected object in each image.

Please note that this simple baseline model is designed to detect only a single occurrence of a class per image. To detect all possible object occurrences in the images, more advanced object detection models like Faster RCNN or YOLO9000 can be explored.

## Multi-Model Network for Segmentation

Additionally, we aim to create a multi-model network for the segmentation task, which will output masks, bounding boxes, and class labels.

## Repository Structure

The repository is organized as follows:

```
- data/                      # Data directory (to be populated with PASCAL VOC 2012 dataset)
- models/                    # Directory to save trained models
- notebooks/                 # Jupyter notebooks for data processing and training
- src/                       # Source code for the neural network models
- README.md                  # This README file
```

## Note

Please refer to the provided Jupyter notebooks in the `notebooks` directory for step-by-step implementation and training details.
