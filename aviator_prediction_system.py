import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AviatorPredictionSystem:
    """
    A comprehensive prediction system for the Aviator game.
    Uses machine learning models to predict outcomes based on historical data.
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the prediction system with the dataset.
        
        Args:
            dataset_path (str): Path to the aviator dataset CSV file
        """
        self.dataset_path = dataset_path
        self.data = None
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = ['color', 'mean', 'var', 'next_approximate']
        self.target_column = 'target'
        
    def load_data(self):
        """Load the dataset from CSV file."""
        try:
            self.data = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded successfully. Shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def explore_data(self, show_visualizations=False):
        """Perform exploratory data analysis."""
        if self.data is None:
            print("Data not loaded. Please load data first.")
            return

        print("\n=== Data Exploration ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Dataset info:")
        print(self.data.info())
        print(f"\nFirst 5 rows:")
        print(self.data.head())
        print(f"\nStatistical summary:")
        print(self.data.describe())
        print(f"\nMissing values:")
        print(self.data.isnull().sum())

        # Visualizations (only if explicitly requested)
        if show_visualizations:
            # Visualizations
            plt.figure(figsize=(15, 10))

            # Distribution of target values
            plt.subplot(2, 3, 1)
            plt.hist(self.data[self.target_column], bins=50, edgecolor='black')
            plt.title('Distribution of Target Values')
            plt.xlabel('Target Value')
            plt.ylabel('Frequency')

            # Correlation heatmap
            plt.subplot(2, 3, 2)
            correlation_matrix = self.data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')

            # Scatter plot: mean vs target
            plt.subplot(2, 3, 3)
            plt.scatter(self.data['mean'], self.data[self.target_column], alpha=0.5)
            plt.title('Mean vs Target')
            plt.xlabel('Mean')
            plt.ylabel('Target')

            # Scatter plot: var vs target
            plt.subplot(2, 3, 4)
            plt.scatter(self.data['var'], self.data[self.target_column], alpha=0.5)
            plt.title('Variance vs Target')
            plt.xlabel('Variance')
            plt.ylabel('Target')

            # Scatter plot: next_approximate vs target
            plt.subplot(2, 3, 5)
            plt.scatter(self.data['next_approximate'], self.data[self.target_column], alpha=0.5)
            plt.title('Next Approximate vs Target')
            plt.xlabel('Next Approximate')
            plt.ylabel('Target')

            # Color distribution
            plt.subplot(2, 3, 6)
            color_counts = self.data['color'].value_counts()
            plt.pie(color_counts.values, labels=color_counts.index, autopct='%1.1f%%')
            plt.title('Color Distribution')

            plt.tight_layout()
            plt.show()
    
    def preprocess_data(self):
        """Preprocess the data for training."""
        if self.data is None:
            print("Data not loaded. Please load data first.")
            return False
        
        # Check for missing values
        if self.data.isnull().sum().sum() > 0:
            print("Found missing values. Filling with median values.")
            self.data = self.data.fillna(self.data.median())
        
        # Separate features and target
        X = self.data[self.feature_columns]
        y = self.data[self.target_column]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        return True
    
    def train_models(self):
        """Train multiple models and compare their performance."""
        if not hasattr(self, 'X_train'):
            print("Data not preprocessed. Please preprocess data first.")
            return False

        # Define models to train
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Linear Regression': LinearRegression()
        }

        print("\n=== Training Models ===")

        for name, model in models.items():
            print(f"Training {name}...")

            # Use scaled data for linear regression, original for tree-based models
            if name == 'Linear Regression':
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            # Store model and metrics
            self.models[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }

            print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Select best model based on R² score
        best_r2 = -float('inf')
        for name, model_data in self.models.items():
            if model_data['r2'] > best_r2:
                best_r2 = model_data['r2']
                self.best_model = model_data['model']
                self.best_model_name = name

        print(f"\nBest model: {self.best_model_name} with R²: {best_r2:.4f}")
        return True

    def load_or_train_model(self, model_path="aviator_prediction_model.pkl"):
        """Load a pre-trained model if it exists, otherwise train a new one."""
        import os
        import joblib

        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}...")
            try:
                model_data = joblib.load(model_path)
                self.best_model = model_data['model']
                self.scaler = model_data['scaler']
                self.best_model_name = model_data['best_model_name']
                self.feature_columns = model_data['feature_columns']

                # Load the dataset to get the data for evaluation
                if self.data is None:
                    self.load_data()

                # Preprocess data to set up test/train sets for evaluation
                if not hasattr(self, 'X_train'):
                    self.preprocess_data()

                # Create a basic models dictionary for the loaded model
                # This is needed for evaluation methods to work properly
                if self.best_model_name and hasattr(self, 'X_test'):
                    # Make predictions to populate the models dict for evaluation
                    if self.best_model_name == 'Linear Regression':
                        y_pred = self.best_model.predict(self.X_test_scaled)
                    else:
                        y_pred = self.best_model.predict(self.X_test)

                    mse = mean_squared_error(self.y_test, y_pred)
                    mae = mean_absolute_error(self.y_test, y_pred)
                    r2 = r2_score(self.y_test, y_pred)

                    self.models = {
                        self.best_model_name: {
                            'model': self.best_model,
                            'mse': mse,
                            'mae': mae,
                            'r2': r2,
                            'predictions': y_pred
                        }
                    }

                print(f"Model loaded successfully from {model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Training a new model instead...")
        else:
            print("No existing model found. Training a new model...")

        # If we get here, we need to train a new model
        if self.data is None:
            self.load_data()

        if not hasattr(self, 'X_train'):
            self.preprocess_data()

        return self.train_models()

    def evaluate_models(self, show_visualizations=False):
        """Evaluate and visualize model performance."""
        if not self.models:
            print("No models trained. Please train models first.")
            return False

        print("\n=== Model Evaluation ===")

        # Print comparison table
        print(f"{'Model':<20} {'MSE':<10} {'MAE':<10} {'R²':<10}")
        print("-" * 55)
        for name, model_data in self.models.items():
            print(f"{name:<20} {model_data['mse']:<10.4f} {model_data['mae']:<10.4f} {model_data['r2']:<10.4f}")

        # Visualization (only if explicitly requested)
        if show_visualizations:
            plt.figure(figsize=(15, 10))

            # Actual vs Predicted for best model
            plt.subplot(2, 3, 1)
            best_predictions = self.models[self.best_model_name]['predictions']
            plt.scatter(self.y_test, best_predictions, alpha=0.5)
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted - {self.best_model_name}')

            # Model comparison - R² scores
            plt.subplot(2, 3, 2)
            model_names = list(self.models.keys())
            r2_scores = [self.models[name]['r2'] for name in model_names]
            plt.bar(model_names, r2_scores)
            plt.title('Model Comparison - R² Score')
            plt.ylabel('R² Score')
            plt.xticks(rotation=45)

            # Model comparison - MAE
            plt.subplot(2, 3, 3)
            mae_scores = [self.models[name]['mae'] for name in model_names]
            plt.bar(model_names, mae_scores)
            plt.title('Model Comparison - MAE')
            plt.ylabel('Mean Absolute Error')
            plt.xticks(rotation=45)

            # Model comparison - MSE
            plt.subplot(2, 3, 4)
            mse_scores = [self.models[name]['mse'] for name in model_names]
            plt.bar(model_names, mse_scores)
            plt.title('Model Comparison - MSE')
            plt.ylabel('Mean Squared Error')
            plt.xticks(rotation=45)

            # Residuals plot for best model
            plt.subplot(2, 3, 5)
            residuals = self.y_test - best_predictions
            plt.scatter(best_predictions, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'Residuals Plot - {self.best_model_name}')

            # Feature importance for best model (if available)
            plt.subplot(2, 3, 6)
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
                plt.bar(self.feature_columns, importances)
                plt.title(f'Feature Importance - {self.best_model_name}')
                plt.ylabel('Importance')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'Feature importance\nnot available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes)
                plt.title(f'Feature Importance - {self.best_model_name}')

            plt.tight_layout()
            plt.show()

        return True
    
    def predict_next_round(self, color, mean, var, next_approximate):
        """
        Predict the outcome for the next round based on input features.
        
        Args:
            color (int): Color value (0 or 1)
            mean (float): Mean value
            var (float): Variance value
            next_approximate (float): Next approximate value
        
        Returns:
            float: Predicted target value
        """
        if self.best_model is None:
            print("No model trained. Please train models first.")
            return None
        
        # Prepare input data
        input_data = np.array([[color, mean, var, next_approximate]])
        
        # Use scaled data for linear regression, original for other models
        if self.best_model_name == 'Linear Regression':
            input_data = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.best_model.predict(input_data)[0]
        
        return max(0, prediction)  # Ensure prediction is non-negative
    
    def simulate_game_round(self, prediction, bet_amount=1.0, multiplier_threshold=2.0):
        """
        Simulate a game round based on prediction.
        
        Args:
            prediction (float): Predicted multiplier
            bet_amount (float): Amount bet
            multiplier_threshold (float): Threshold for cashing out
        
        Returns:
            dict: Simulation results
        """
        actual_multiplier = prediction  # In real scenario, this would come from the actual game
        
        # Determine if we should cash out based on prediction
        should_cash_out = actual_multiplier >= multiplier_threshold
        
        if should_cash_out:
            cash_out_multiplier = multiplier_threshold
            payout = bet_amount * cash_out_multiplier
            profit = payout - bet_amount
        else:
            # If we didn't cash out in time, we lose the bet
            cash_out_multiplier = 0
            payout = 0
            profit = -bet_amount
        
        return {
            'predicted_multiplier': prediction,
            'actual_multiplier': actual_multiplier,
            'cash_out_threshold': multiplier_threshold,
            'should_cash_out': should_cash_out,
            'cash_out_multiplier': cash_out_multiplier,
            'bet_amount': bet_amount,
            'payout': payout,
            'profit': profit
        }
    
    def run_prediction_analysis(self, num_rounds=10):
        """
        Run prediction analysis for multiple rounds.
        
        Args:
            num_rounds (int): Number of rounds to analyze
        """
        if self.data is None:
            print("Data not loaded. Please load data first.")
            return
        
        print(f"\n=== Running Prediction Analysis for {num_rounds} Rounds ===")
        
        # Sample some recent rounds from the dataset for demonstration
        recent_data = self.data.tail(num_rounds * 2)  # Get more than needed to have variety
        
        results = []
        total_profit = 0
        
        for i in range(min(num_rounds, len(recent_data))):
            row = recent_data.iloc[i]
            
            # Make prediction based on features
            prediction = self.predict_next_round(
                row['color'], 
                row['mean'], 
                row['var'], 
                row['next_approximate']
            )
            
            if prediction is not None:
                # Simulate the round
                simulation = self.simulate_game_round(
                    prediction=prediction,
                    bet_amount=1.0,
                    multiplier_threshold=2.0
                )
                
                # Calculate accuracy
                actual = row[self.target_column]
                accuracy = 1 - abs(prediction - actual) / max(actual, 0.001)  # Avoid division by zero
                
                results.append({
                    'round': i+1,
                    'predicted': prediction,
                    'actual': actual,
                    'accuracy': max(0, accuracy),  # Clamp to 0-1 range
                    'profit': simulation['profit']
                })
                
                total_profit += simulation['profit']
        
        # Print results
        print(f"{'Round':<6} {'Predicted':<10} {'Actual':<10} {'Accuracy':<10} {'Profit':<10}")
        print("-" * 50)
        
        for result in results:
            print(f"{result['round']:<6} {result['predicted']:<10.3f} {result['actual']:<10.3f} {result['accuracy']:<10.3f} {result['profit']:<10.3f}")
        
        print(f"\nTotal Profit over {len(results)} rounds: {total_profit:.3f}")
        if len(results) > 0:
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
            print(f"Average Prediction Accuracy: {avg_accuracy:.3f}")
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        import joblib
        if self.best_model is None:
            print("No model trained to save.")
            return False
        
        try:
            joblib.dump({
                'model': self.best_model,
                'scaler': self.scaler,
                'best_model_name': self.best_model_name,
                'feature_columns': self.feature_columns
            }, filepath)
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load a trained model from a file."""
        import joblib
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.best_model_name = model_data['best_model_name']
            self.feature_columns = model_data['feature_columns']
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    """Main function to run the Aviator prediction system."""
    print("=== Aviator Game Prediction System ===")

    # Initialize the system
    dataset_path = "aviator_dataset_clean.csv"
    system = AviatorPredictionSystem(dataset_path)

    # Load or train model
    if not system.load_or_train_model("aviator_prediction_model.pkl"):
        print("Failed to load or train model. Exiting.")
        return

    # Load data if not already loaded during model loading/training
    if system.data is None:
        if not system.load_data():
            return

    # Explore data (without visualizations)
    system.explore_data(show_visualizations=False)

    # Evaluate models (this will work with loaded or newly trained models) - without visualizations
    system.evaluate_models(show_visualizations=False)

    # Run prediction analysis
    system.run_prediction_analysis(num_rounds=20)

    # Example of making a single prediction
    print("\n=== Some Prediction ===")
    example_prediction = system.predict_next_round(
        color=1,
        mean=2.5,
        var=1.2,
        next_approximate=3.7
    )
    print(f"Prediction for some input: {example_prediction:.3f}")

    # Simulate a game round
    simulation_result = system.simulate_game_round(
        prediction=example_prediction,
        bet_amount=5.0,
        multiplier_threshold=2.5
    )
    print(f"Simulation result: {simulation_result}")

    # Save the model (in case it was newly trained)
    if system.best_model is not None:
        system.save_model("aviator_prediction_model.pkl")

    print("\n=== System Completed ===")

if __name__ == "__main__":
    main()