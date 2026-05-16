# neural_project
# 🏠 House Price Prediction using MLP Neural Network

## Overview
This project predicts house prices using a Multilayer Perceptron (MLP) built with PyTorch.

---

## Dataset
California Housing Dataset from sklearn.

---

## Model
- MLP Neural Network
- Flexible hidden layers
- Baseline Architecture: 64 → 32
- Deep Architecture: 128 → 64 → 32
- Activation Function: ReLU
- Batch Normalization
- Dropout Regularization
- Output Layer: 1 neuron (Regression)

---

## Training
- Loss Function: MSELoss
- Optimizer: Adam
- Early Stopping used to reduce overfitting

---

## Preprocessing
- StandardScaler normalization
- Train / Validation / Test split

---

## Experiments

| Experiment | Architecture | Learning Rate |
|------------|-------------|----------------|
| Exp1 | 64-32 | 0.001 |
| Exp2 | 128-64-32 | 0.001 |
| Exp3 | 64-32 | 0.01 |

---

## Results
- Deep architecture achieved better performance
- High learning rate caused less stable training
- Early stopping helped reduce overfitting

---

## Visualization
The project includes:
- Training Loss curves
- Validation Loss curves
- Comparison plots between experiments

---

## Run Project

```bash
pip install torch scikit-learn matplotlib
python train.py
```
