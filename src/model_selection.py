import pandas as pd
import json
from segmentation_model import train_segmentation_model
from prediction_model import train_prediction_model
from save_results import save_metrics

def load_config():
    with open("C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/configurations.json", "r") as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    config = load_config()
    
    # Load processed data for segmentation and prediction
    data = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/data/encoded_customer_sales_data.csv")
    
    # Train segmentation model and save results
    segmented_data, seg_model = train_segmentation_model(data, config["segmentation_model"])
    segmented_data.to_csv("C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/segmentation_results.csv", index=False)
    
    # Train prediction model using the segmented data
    y_test, predictions, pred_model = train_prediction_model(segmented_data, config["prediction_model"])
    
    # Save model metrics (e.g., RMSE, R², etc.)
    save_metrics(y_test, predictions)

    print(segmented_data.head())  # After segmentation
print(y_test.head())          # After splitting for prediction
