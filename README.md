# Aviator Game Prediction System

This project implements a machine learning-based prediction system for the Aviator game. It uses historical game data to predict outcomes for each round and provides insights for better decision-making.

## Features

- Multiple machine learning models for prediction (Random Forest, Gradient Boosting, Linear Regression)
- Comprehensive data exploration and visualization
- Model comparison and evaluation
- Prediction accuracy assessment
- Game simulation with configurable parameters
- Model persistence for future use

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## Installation

1. Clone or download this repository
2. Open in VS studio, open Terminal and create virtual environment Using below command:
      python -m venv .venv
3.     Activate Virtual Environment:
      .venv/Scripts/activate
4. Install required packages:
   pip install -r requirements.txt
   ```

## Usage

1. Place your dataset file `aviator_dataset_clean.csv` in the project directory
2. Run the prediction system:
   ```bash
   python run_prediction.py
   ```

## Files

- `aviator_prediction_system.py`: Main prediction system implementation
- `run_prediction.py`: Script to run the prediction system
- `config.json`: Configuration settings
- `requirements.txt`: Required Python packages
- `aviator_dataset_clean.csv`: Input dataset (you need to provide this)

## How It Works

1. **Data Loading**: The system loads the aviator dataset containing features like color, mean, variance, and next approximate values
2. **Exploration**: Performs exploratory data analysis with visualizations
3. **Preprocessing**: Splits data into training and testing sets, scales features
4. **Model Training**: Trains multiple models and compares their performance
5. **Evaluation**: Evaluates models using various metrics (MSE, MAE, R²)
6. **Prediction**: Uses the best model to predict outcomes for new rounds
7. **Simulation**: Simulates game rounds based on predictions

## Model Performance Metrics

- **MSE (Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² (R-squared)**

## Prediction Example

The system can predict the multiplier for the next round based on input features:
- Color: Indicator of game state
- Mean: Average value from previous rounds
- Variance: Variability in previous rounds
- Next Approximate: Estimated next value

## Risk Management

The system includes a simulation feature that allows you to:
- Set cash-out thresholds
- Configure bet amounts
- Evaluate potential profits/losses

## Model Persistence

Trained models can be saved and loaded for future use:
- Save: `system.save_model("model.pkl")`
- Load: `system.load_model("model.pkl")`

## Customization

You can customize the system by modifying:
- `config.json`: Change model parameters, thresholds, etc.
- `aviator_prediction_system.py`: Add new models, features, or metrics