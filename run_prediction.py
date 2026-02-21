import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aviator_prediction_system import AviatorPredictionSystem

def run_prediction_system():
    """Run the Aviator prediction system."""
    print("Starting Aviator Prediction System...")

    # Initialize the system
    dataset_path = "aviator_dataset_clean.csv"
    system = AviatorPredictionSystem(dataset_path)

    # Load or train model
    print("\nLoading or training model...")
    if not system.load_or_train_model("aviator_prediction_model.pkl"):
        print("Failed to load or train model. Exiting.")
        return

    # Load data if not already loaded during model loading/training
    if system.data is None:
        if not system.load_data():
            print("Failed to load data. Exiting.")
            return

    # Explore data (without visualizations)
    print("\nExploring data...")
    system.explore_data(show_visualizations=False)

    # Evaluate models (without visualizations)
    print("\nEvaluating models...")
    system.evaluate_models(show_visualizations=False)

    # Run prediction analysis
    print("\nRunning prediction analysis...")
    system.run_prediction_analysis(num_rounds=20)

    # Example of making a single prediction
    print("\nMaking example prediction...")
    example_prediction = system.predict_next_round(
        color=1,
        mean=2.5,
        var=1.2,
        next_approximate=3.7
    )
    print(f"Example prediction: {example_prediction:.3f}")

    # Simulate a game round
    simulation_result = system.simulate_game_round(
        prediction=example_prediction,
        bet_amount=5.0,
        multiplier_threshold=2.5
    )
    print(f"Simulation result: {simulation_result}")

    # Save the model (in case it was newly trained)
    print("\nSaving model...")
    if system.best_model is not None:
        system.save_model("aviator_prediction_model.pkl")

    print("\nPrediction system completed successfully!")

if __name__ == "__main__":
    run_prediction_system()