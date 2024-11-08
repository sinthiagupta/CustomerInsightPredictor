import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np  # For calculating square root

def save_metrics(y_test, predictions):
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Calculate R-squared
    r2 = r2_score(y_test, predictions)
    
    # Create a DataFrame to store the metrics
    metrics = pd.DataFrame({
        "Metric": ["Mean Squared Error", "Root Mean Squared Error", "R-squared"],
        "Value": [mse, rmse, r2]
    })
    
    # Save metrics to a CSV file
    metrics.to_csv("C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/model_metrics.csv", index=False)
    
    # Save metrics summary to a text file
    with open("C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/results_summary.txt", "w") as file:
        file.write(f"Mean Squared Error: {mse}\n")
        file.write(f"Root Mean Squared Error: {rmse}\n")
        file.write(f"R-squared: {r2}\n")

if __name__ == "__main__":
    y_test = [1, 2, 3]  # Example true values
    predictions = [1.1, 2.1, 2.9]  # Example predicted values
    save_metrics(y_test, predictions)
