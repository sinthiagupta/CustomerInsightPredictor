import pandas as pd
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def load_configuration(config_file=None):
    """Function to load configuration from a JSON file if provided."""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            print("Loaded Configuration:", config)
            return config
    else:
        # Default configuration if no file provided
        config = {
            "input_file": "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/segmentation_results.csv",
            "output_file": "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/prediction_result.csv",
            "segmentation_model": {
                "algorithm": "KMeans",
                "n_clusters": 8,
                "random_state": 42
            },
            "prediction_model": {
                "algorithm": "LinearRegression",
                "parameters": {
                    "fit_intercept": True
                }
            }
        }
        print("Using default configuration:", config)
        return config


def perform_prediction(config):
    """Function to perform prediction on data."""
    try:
        # Load the dataset
        input_file = config["input_file"]
        df = pd.read_csv(input_file)

        target_column = 'Sales Amount (k$)'  # Column you are predicting

        # Ensure the target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        
        # Separate features (X) and target (y)
        X = df.drop(columns=[target_column])  # Features (all columns except target)
        y = df[target_column]  # Target column (e.g., Sales)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the prediction model
        model_name = config["prediction_model"]["algorithm"]
        if model_name == "LinearRegression":
            model = LinearRegression(**config["prediction_model"]["parameters"])
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        model.fit(X_train, y_train)

        # Predict the target variable on the test set
        predictions = model.predict(X_test)

        # Combine actual and predicted data into a DataFrame
        results_df = pd.DataFrame({
            'Actual': y_test,      # Actual target values
            'Predicted': predictions  # Predicted target values
        })

        # Save the results to the output file
        output_file = config["output_file"]
        results_df.to_csv(output_file, index=False)

        print("Prediction completed successfully. Results saved to:", output_file)

    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    config_file = "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/configurations.json"  # Path to your config file
    config = load_configuration(config_file)  # Load configuration from the file

    # Ensure perform_prediction is called correctly
    perform_prediction(config)  # Pass config as the only argument
