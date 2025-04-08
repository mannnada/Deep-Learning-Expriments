# Deep Learning Experiments

This repository contains a collection of deep learning experiments for regression, classification, and image classification tasks using various datasets. Each experiment includes model architectures, training scripts, and evaluation metrics.

---

## 1. Regression: California Housing Dataset

### Problem
The objective of this task is to predict housing prices using the **California Housing dataset**. The model estimates the median house value based on features such as:
- Median income
- Average number of rooms
- Latitude & longitude
- Population density, etc.

### Model Architecture
A **Multilayer Perceptron (MLP)** architecture is used for the regression task:
- **Input Layer**: Accepts numerical housing features
- **Hidden Layers**: Several Dense layers with ReLU activation, BatchNormalization, and Dropout for regularization
- **Output Layer**: A single neuron for predicting the target variable

### Artifacts
- **Training History Plots**: Visualizations of loss and performance over epochs
- **Model Visualization**: Diagram of the MLP architecture
- **Regression Metrics**: MSE, RMSE, MAE, and additional evaluation metrics
- **Error Analysis**: Residual plots and error distribution analysis

📌 **Colab Notebook**: https://github.com/mannnada/Deep-Learning-Expriments/blob/main/README.md
---

## 2. Classification: Wildfire Prediction

### Problem
The goal is to predict wildfire occurrences using **sensor data**. The model classifies whether a wildfire will occur (1 = Wildfire, 0 = No Wildfire) based on environmental conditions such as:
- Temperature
- Humidity
- Wind speed
- Vegetation indices

### Model Architecture
A **Multilayer Perceptron (MLP)** is used for binary classification:
- **Input Layer**: Accepts environmental sensor data
- **Hidden Layers**: Multiple Dense layers with ReLU activation, Dropout, and BatchNormalization
- **Output Layer**: A single neuron with **sigmoid activation** for binary classification

📌 **Colab Notebook**: https://colab.research.google.com/drive/1122KXteYdByTKDuCWmNCznpHH2uofs0E?usp=sharing
---

## 3. Image Classification: Fashion MNIST

### Problem
The objective is to classify fashion items from the **Fashion MNIST dataset**. The model categorizes images into one of **10 clothing classes**, including T-shirts, trousers, dresses, and sneakers.

### Model Architecture
A **Convolutional Neural Network (CNN)** is used for image classification:
- **Input Layer**: Accepts 28x28 grayscale images
- **Convolutional Layers**: Several Conv2D layers with ReLU activation and MaxPooling for feature extraction
- **Fully Connected Layers**: Flattened layer followed by Dense layers for feature learning
- **Output Layer**: A **softmax layer** with 10 neurons for multi-class classification

### Artifacts
- **Training History Plots**: Visualization of loss and accuracy over epochs
- **Model Visualization**: Diagram of the CNN architecture
- **Classification Metrics**: Accuracy, Precision, Recall, and F1-score (both overall and per class)
- **Confusion Matrix**: Heatmap to analyze misclassifications per class
- **ROC & PR Curves**: Performance evaluation using Receiver Operating Characteristic and Precision-Recall curves
- **Error Analysis**: Per-class analysis with sample misclassified images

📌 **Colab Notebook**: https://colab.research.google.com/drive/14FGNQMFkZZqgXxmhJmzMqMPLBCG4tgQJ?usp=sharing

📺 **YouTube Walkthrough**: 

---

## 📊 Experiment Tracking & Monitoring
This repository integrates **Weights & Biases (wandb)** for logging training metrics, model visualizations, and evaluation metrics. This allows for:
- Live tracking of training progress
- Hyperparameter tuning
- Visualization of model performance and errors
