import json

def load_config(file_path="C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/configurations.json"):
    with open(file_path, "r") as file:
        return json.load(file)
