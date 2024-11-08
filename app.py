from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import json
from src.segmentation_model import train_segmentation_model
from src.prediction_model import train_prediction_model
from src.save_results import save_metrics
import os  # Add this line

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))



# Load configuration
def load_config():
    config_path = "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/configurations.json"  # Path to configurations.json
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle segmentation
@app.route('/segment', methods=['POST'])
def segment():
    config = load_config()
    segmentation_config = config["segmentation_model"]  # Get segmentation model config

    # Path to processed data
    data_path = "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/data/encoded_customer_sales_data.csv"
    data = pd.read_csv(data_path)
    
    # Run segmentation model
    segmented_data, seg_model = train_segmentation_model(data, segmentation_config)
    
    # Path to save segmentation results
    segmentation_results_path = "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/segmentation_results.csv"
    segmented_data.to_csv(segmentation_results_path, index=False)

    return render_template('index.html', segment_success=True)

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    config = load_config()
    prediction_config = config["prediction_model"]  # Get prediction model config

    # Load segmentation results for prediction model
    segmented_data_path = "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/segmentation_results.csv"
    data = pd.read_csv(segmented_data_path)
    
    # Run prediction model
    y_test, predictions, pred_model = train_prediction_model(data, prediction_config)

    # Save prediction metrics
    save_metrics(y_test, predictions)

    # Load metrics for display
    metrics_path = "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/model_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    mse = metrics.loc[metrics["Metric"] == "Mean Squared Error", "Value"].values[0]
    rmse = metrics.loc[metrics["Metric"] == "Root Mean Squared Error", "Value"].values[0]
    r2 = metrics.loc[metrics["Metric"] == "R-squared", "Value"].values[0]

    return render_template('index.html', predict_success=True, mse=mse, rmse=rmse, r2=r2)

if __name__ == "__main__":
    app.run(debug=True)
