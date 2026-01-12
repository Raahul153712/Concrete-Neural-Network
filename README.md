# Concrete-Neural-Network
Using an MLP to predict the concrete compressive strength.
# Concrete Compressive Strength Prediction (MLP)

This repository implements a Multi-Layer Perceptron (MLP) using **PyTorch** to predict the compressive strength of concrete based on its composition and age. The project focuses on model stability and error analysis by using a multi-run experiment setup and robust visualization techniques.

## 📊 Project Overview
Concrete strength is determined by complex, non-linear interactions between its ingredients. This project utilizes a deep neural network to learn these patterns from the [Concrete Compressive Strength Dataset](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength).

### Key Features:
- **Custom Early Stopping:** Monitors validation loss and restores the best model weights to prevent overfitting.
- **Experimental Loop:** Runs the model multiple times (3 runs) to ensure results are consistent and not due to lucky weight initialization.
- **Robust Evaluation:** Includes MAE, MSE, RMSE, and RMSLE metrics.
- **Advanced Visualization:** 
  - **Accuracy by Range:** Analyzes how the model performs at different strength levels (e.g., 0-20 MPa vs 60+ MPa).
  - **Q-Q Plots:** Checks the distribution of residuals to evaluate prediction bias.

## 🧬 Model Architecture
The network is a deep regressor built with the following structure:
- **Input Layer:** 8 features (Cement, Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age).
- **Hidden Layers:** 5 fully connected layers with **LeakyReLU** activations:
  - Layer 1: 10 neurons
  - Layer 2: 50 neurons
  - Layer 3: 100 neurons
  - Layer 4: 100 neurons
  - Layer 5: 100 neurons
- **Output Layer:** 1 neuron (Predicted Strength).
- **Optimizer:** Adam (Learning Rate = 0.001).

## 🛠️ Requirements
To run the code, you will need:
- Python 3.x
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SciPy

```bash
pip install torch pandas numpy matplotlib scikit-learn scipy
