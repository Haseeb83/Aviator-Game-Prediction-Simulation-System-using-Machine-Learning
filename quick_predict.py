#!/usr/bin/env python3
"""
Simple prediction script that loads the existing model and makes predictions
without generating graphs or extensive data exploration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aviator_prediction_system import AviatorPredictionSystem

def quick_prediction(color=1, mean=2.5, var=1.2, next_approximate=3.7):
    """Make a quick prediction using the existing model."""
    print("=== Quick Aviator Prediction ===")
    
    # Initialize the system
    dataset_path = "aviator_dataset_clean.csv"
    system = AviatorPredictionSystem(dataset_path)
    
    # Load existing model (or train a new one if it doesn't exist)
    print("Loading model...")
    success = system.load_or_train_model("aviator_prediction_model.pkl")
    
    if not success:
        print("Failed to load or train model.")
        return None
    
    print(f"Using model: {system.best_model_name}")
    
    # Make prediction
    print(f"\nMaking prediction for:")
    print(f"  Color: {color}")
    print(f"  Mean: {mean}")
    print(f"  Variance: {var}")
    print(f"  Next Approximate: {next_approximate}")
    
    prediction = system.predict_next_round(color, mean, var, next_approximate)
    
    print(f"\nPrediction: {prediction:.3f}")
    
    # Simulate a game round
    simulation_result = system.simulate_game_round(
        prediction=prediction,
        bet_amount=5.0,
        multiplier_threshold=2.5
    )
    
    print(f"\nSimulation result if you bet $5.00:")
    print(f"  Predicted multiplier: {prediction:.3f}")
    print(f"  Cash out threshold: {simulation_result['cash_out_threshold']}")
    print(f"  Should cash out: {simulation_result['should_cash_out']}")
    print(f"  Payout: ${simulation_result['payout']:.2f}")
    print(f"  Profit: ${simulation_result['profit']:.2f}")
    
    return prediction

def batch_predictions(predictions_data):
    """Make multiple predictions at once."""
    print("\n=== Batch Predictions ===")
    
    # Initialize the system
    dataset_path = "aviator_dataset_clean.csv"
    system = AviatorPredictionSystem(dataset_path)
    
    # Load existing model (or train a new one if it doesn't exist)
    print("Loading model...")
    success = system.load_or_train_model("aviator_prediction_model.pkl")
    
    if not success:
        print("Failed to load or train model.")
        return None
    
    print(f"Using model: {system.best_model_name}")
    print(f"\nMaking {len(predictions_data)} predictions:\n")
    
    results = []
    for i, data in enumerate(predictions_data):
        color, mean, var, next_approximate = data
        prediction = system.predict_next_round(color, mean, var, next_approximate)
        results.append(prediction)
        
        print(f"Prediction {i+1}: {prediction:.3f} (input: color={color}, mean={mean}, var={var}, next_approx={next_approximate})")
    
    return results

if __name__ == "__main__":
    # Example 1: Single prediction
    quick_prediction()
    
    # Example 2: Multiple predictions
    sample_data = [
        (1, 2.5, 1.2, 3.7),
        (0, 1.8, 0.9, 2.1),
        (1, 3.2, 1.5, 4.0),
        (0, 2.1, 1.1, 2.8),
        (1, 4.0, 2.0, 5.5)
    ]
    
    batch_results = batch_predictions(sample_data)