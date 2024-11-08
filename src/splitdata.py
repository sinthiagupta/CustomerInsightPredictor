import pandas as pd
from sklearn.model_selection import train_test_split

# Load the preprocessed data
data = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\CustomerInsightPredictor\\data\\scaled_customer_data.csv')  # Update with actual file path

# Define features (X) and target (y)
X = data.drop(columns=['Sales Amount (k$)'])  # Replace with the target column name
y = data['Sales Amount (k$)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split datasets
X_train.to_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\CustomerInsightPredictor\\data\\X_train.csv.csv', index=False)
X_test.to_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\CustomerInsightPredictor\\data\\X_test.csv.csv', index=False)
y_train.to_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\CustomerInsightPredictor\\data\\y_train.csv.csv', index=False)
y_test.to_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\CustomerInsightPredictor\\data\\y_test.csv.csv', index=False)

# Output the shapes of the datasets
print(f"Training set features size: {X_train.shape[0]} samples")
print(f"Testing set features size: {X_test.shape[0]} samples")
print(f"Training set target size: {y_train.shape[0]} samples")
print(f"Testing set target size: {y_test.shape[0]} samples")


print("Data splitting completed and saved to CSV files.")
