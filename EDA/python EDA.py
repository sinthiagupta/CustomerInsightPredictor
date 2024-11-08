# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your combined dataset
data = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\CustomerInsightPredictor\\data\\combined_customer_sales_data.csv')


'''# Display the first few rows of the data
print(data.head())

# Check basic information
print(data.info())

# Get statistical summary
print(data.describe())'''

#Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

'''# Visualize missing values
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()
'''
'''
# Histograms for numerical columns
data.hist(bins=20, figsize=(15, 10))
plt.suptitle("Numerical Feature Distributions")
plt.show()

# Box plot to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=data.select_dtypes(include=['float64', 'int64']))
plt.title("Box Plot for Outlier Detection")
plt.show()'''

'''# Bar plots for categorical data (like 'Gender')
plt.figure(figsize=(8, 4))
sns.countplot_x(x='Gender_x', data=data)
plt.title("Gender Distribution")
plt.show()'''

'''# Correlation heatmap
numeric_data = data.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()'''

'''data.columns = data.columns.str.strip()  
print(data.columns)  

income_spending_data = pd.DataFrame({
    'CustomerID': data['CustomerID'].repeat(2),
    'Gender': pd.concat([data['Gender_x'], data['Gender_y']]),
    'Annual Income (k$)': pd.concat([data['Annual Income (k$)_x'], data['Annual Income (k$)_y']]),
    'Spending Score (1-100)': pd.concat([data['Spending Score (1-100)_x'], data['Spending Score (1-100)_y']]),
    'Sales Amount (k$)': pd.concat([data['Sales Amount (k$)']] * 2)
})

# Reset index for the new DataFrame
income_spending_data.reset_index(drop=True, inplace=True)

# Display the first few rows of the new DataFrame
print(income_spending_data.head())


# Scatter plot for Sales vs. Annual Income
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Sales Amount (k$)', data=income_spending_data, hue='Gender', alpha=0.7)
plt.title("Sales Amount vs Annual Income by Gender")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Sales Amount (k$)")
plt.legend(title='Gender')
plt.show()

# Box plot to see spending patterns by gender
plt.figure(figsize=(8, 5))
sns.boxplot(x='Gender', y='Spending Score (1-100)', data=income_spending_data)
plt.title("Spending Score by Gender")
plt.xlabel("Gender")
plt.ylabel("Spending Score (1-100)")

plt.tight_layout()
plt.show()'''
