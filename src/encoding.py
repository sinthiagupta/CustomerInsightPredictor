#encoding 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load your dataset
data = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\CustomerInsightPredictor\\data\\preprocessed_customer_data_final.csv')

# Identify categorical columns
categorical_cols = ['Gender_x', 'Gender_y']

# Create an instance of OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')

# Fit and transform the encoder on the selected columns
encoded_columns = encoder.fit_transform(data[categorical_cols])

# Create a DataFrame with the new encoded columns
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate the encoded DataFrame with the original DataFrame
data = pd.concat([data, encoded_df], axis=1)

# Drop the original categorical columns
data.drop(categorical_cols, axis=1, inplace=True)

# Display the first few rows of the updated dataset
print(data.head())


# Save the encoded dataset to a specific directory
data.to_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\CustomerInsightPredictor\\data\\encoded_customer_sales_data.csv', index=False)
