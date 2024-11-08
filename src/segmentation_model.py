from sklearn.cluster import KMeans
import pandas as pd
import json
def load_config():
    config_path = "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/configurations.json"
    with open(config_path, "r") as file:
        config = json.load(file)
    
    print("Loaded configuration:", config)  # Debug print to check config content
    return config


def train_segmentation_model(df, config):
    df = pd.get_dummies(df, drop_first=True)
    
    # Ensure keys exist with a fallback value if not found
    n_clusters = config.get("n_clusters", 3)  # Default to 3 if missing
    random_state = config.get("random_state", 42)

    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    
    clusters = model.fit_predict(df)
    df['Cluster'] = clusters
    
    return df, model


# Example usage
if __name__ == "__main__":
    config = load_config()
    data = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/data/encoded_customer_sales_data.csv")  # Load your dataset
    segmented_data, model = train_segmentation_model(data, config)
    
    # Save segmented data to results folder for later analysis
    segmented_data.to_csv("C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/segmentation_results.csv", index=False)
