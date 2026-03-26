# Binary CAN Intrusion Detection

This project focuses on detecting malicious activity in automotive Controller Area Network (CAN) traffic using a neural network baseline. The main goal is to classify CAN messages as either **normal** or **attack** based on engineered message-level features.

## Project Overview

Modern vehicles rely on the CAN bus for communication between electronic control units (ECUs). Because CAN does not provide built-in authentication or encryption, it is vulnerable to message injection, spoofing, and denial-of-service attacks. This project applies machine learning to build an intrusion detection system for CAN traffic.

## Features

- Binary classification of CAN traffic
- Engineered features from CAN messages
- Data preprocessing and cleaning pipeline
- Neural network baseline using a multilayer perceptron (MLP)
- Evaluation with accuracy, precision, recall, F1-score, and ROC-AUC

## Dataset Features Used

The model was trained using the following input features:

- dlc
- can_id_int
- byte_0_int to byte_7_int
- inter_arrival
- payload_sum
- nonzero_bytes
- payload_unique_values

## Model

The baseline model is a multilayer perceptron (MLP) with:
- Dense layers with ReLU activation
- Dropout for regularization
- Sigmoid output layer for binary classification

## Results

The binary classifier achieved the following validation performance:

- **Accuracy:** 85.91%
- **Precision:** 99.23%
- **Recall:** 85.74%
- **F1-score:** 91.99%
- **ROC-AUC:** 0.9624

## Repository Structure

```bash
data/        # processed datasets
models/      # saved trained models
logs/        # training logs and outputs
notebooks/   # experiments and analysis
src/         # training and preprocessing scripts
