from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import json

def load_config():
    with open("C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/configurations.json", "r") as file:
        config = json.load(file)
    return config

def train_prediction_model(df, config):
    # Update the target column name based on your dataset
    if 'Sales Amount (k$)' not in df.columns:
        raise KeyError("'Sales Amount (k$)' column not found in the DataFrame")
    
    X = df.drop(columns=["Sales Amount (k$)", "Cluster"], errors='ignore')  # Drop the 'Sales Amount' and 'Cluster' columns
    y = df["Sales Amount (k$)"]  # Use the correct target column
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Instantiate and train the model
    model = LinearRegression(**config["parameters"])
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    return y_test, predictions, model

if __name__ == "__main__":
    config = load_config()["prediction_model"]
    data = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/segmentation_results.csv")  # Load data with cluster labels

    # Check the columns of the data to ensure the target column is correct
    print(data.columns)

    try:
        y_test, predictions, model = train_prediction_model(data, config)
        results_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
        results_df.to_csv("C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/predictions.csv", index=False)
    except KeyError as e:
        print(f"Error: {e}")
