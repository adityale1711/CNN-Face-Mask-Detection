# Face Mask Detection with CNN

This project aims to create a deep learning model that can detect whether a person is wearing a face mask or not. The model is built using Convolutional Neural Networks (CNN) and trained on a dataset of images with and without face masks.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Usage](#usage)
- [Results](#results)

## Introduction
Face Mask Detection has become an essential task during the COVID-19 pandemic. This project uses a CNN to classify images into two categories: with a mask and without a mask. The model can be integrated into various applications to ensure safety and compliance with health guidelines.

## Dataset
The dataset used in this project consists of images of people with and without face masks. You can download the dataset from [Kaggle - Face Mask Detection Dataset](https://www.kaggle.com/andrewmvd/face-mask-detection).

## Installation
1. Clone the repository
```bash
git clone https://github.com/adityale1711/CNN-Face-Mask-Detection.git
cd CNN-Face-Mask-Detection
```

2. Create a Conda Environment
```bash
conda create -n envname python=3.9
conda activate envname
```

3. Install Required Packages
```bash
pip install -r requirements.txt
```

## Model Architecture
The CNN model is built using the following architecture and hyperparameter tuning:

1. Model Build with 2D Convolutional Neural Network with input shape (35, 35, 3) where '35, 35' target image size and '3' is RGB channel

2. Hyperparameter Tuned with three options:
- Random Search
- Bayesian Optimization
- Hyperband

3. Tuner parameter lists:
- Number of Feature Extraction layer (2D Convolution) with maximum 5 layers
- Convolution Filter with options: 8, 16, 32 and 64
- Kernel Size with option 3x3 or 5x5
- Stride with option 1 or 3
- Number of Fully Connected layer with maximum 5 layers
- Fully connected units with options: 8, 16, 32 and 64
- Dropout Rate with range between 0.1 to 0.9 and 0.1 steps
- Three option for Optimizer: Adam, RMSProp and SGD
- Loss function with option Binary Crossentropy for two output class or Categorical Crossentropy for more than two class
- Initiate first learning rate before adapted by callback

4. Train model configured with 100 epochs, 256 batch size and 3 callbacks:
- Model Checkpoint to save best weight only with monitoring validation accuracy
- Train will be stopped Early when validation accuracy doesn't improve in 50 epochs
- Learning rate will reduced by factor 0.3 when validation accuracy doesn't improve in 10 epocs

## Training and Evaluation
To train the model, run:
```bash
python python_code/Modeling/train_and_evaluate.py
```
or:
```bash
cd python_code/Modeling
python train_and_evaluate.py
```

- Input model file name
- After training complete, select trained model file (*.h5) for evaluation
- input yes or no for model plot history
- input yes or no for confusion matrix result

## Usage
To use this app, run:
```bash
python python_code/Prediction/predict.py
```
or:
```bash
cd python_code/Prediction
python predict.py
```

- Select model file
- Input 'image' if you want to detect on Image or input 'video' if you want to detect on video
- the detection results will saved to 'python_code/Prediction/results'

## Results
- Image
  ![img_result](https://github.com/adityale1711/CNN-Face-Mask-Detection/assets/72447020/1cee88fd-3f3b-4380-a49a-8a955da36311)

- Video

https://github.com/adityale1711/CNN-Face-Mask-Detection/assets/72447020/30d67b25-1a46-4084-b7db-3b47b885feac

