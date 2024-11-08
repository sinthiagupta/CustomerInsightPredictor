import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load data
data = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\CustomerInsightPredictor\\data\\encoded_customer_sales_data.csv')


# Separate columns by type
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Scale numeric features
scaler = MinMaxScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Final processed data
print("\nFinal processed data:")
print(data.head())

# Save preprocessed data
data.to_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\CustomerInsightPredictor\\data\\scaled_customer_data.csv', index=False)
print("Preprocessing complete. Saved to 'preprocessed_customer_data_final.csv'.")
