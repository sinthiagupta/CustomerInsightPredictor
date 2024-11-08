# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your combined dataset
data = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\CustomerInsightPredictor\\data\\combined_customer_sales_data.csv')


# Display the first few rows of the data
print(data.head())

# Check basic information
print(data.info())

# Get statistical summary
print(data.describe())

#Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Visualize missing values
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Histograms for numerical columns
data.hist(bins=20, figsize=(15, 10))
plt.suptitle("Numerical Feature Distributions")
plt.show()

# Box plot to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=data.select_dtypes(include=['float64', 'int64']))
plt.title("Box Plot for Outlier Detection")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# First plot for 'Gender_x'
sns.countplot(x='Gender_x', data=data, ax=axes[0])
axes[0].set_title("Gender_x Distribution")

# Second plot for 'Gender_y'
sns.countplot(x='Gender_y', data=data, ax=axes[1])
axes[1].set_title("Gender_y Distribution")

# Show the plot
plt.tight_layout()
plt.show()


# Correlation heatmap
numeric_data = data.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

data.columns = data.columns.str.strip()  # Strip any leading/trailing spaces from column names

# Print column names to verify
print(data.columns)

# Reshape the DataFrame using pd.melt()
# Melt the dataset so we can combine Gender, Income, and Spending Score properly for both x and y
income_spending_data = pd.melt(data,
                               id_vars=['CustomerID', 'Sales Amount (k$)'],
                               value_vars=['Gender_x', 'Gender_y', 'Annual Income (k$)_x', 'Annual Income (k$)_y', 
                                            'Spending Score (1-100)_x', 'Spending Score (1-100)_y'],
                               var_name='Variable', value_name='Value')

# Now we need to separate the Gender and Income/Spending Score variables
# Extract the relevant parts (Gender, Income, Spending Score) from the 'Variable' column
income_spending_data['Attribute'] = income_spending_data['Variable'].str.extract(r'(.+)_(x|y)')[0]
income_spending_data['Type'] = income_spending_data['Variable'].str.extract(r'(.+)_(x|y)')[1]

# Now, pivot the data into a more usable format
income_spending_data_pivot = income_spending_data.pivot_table(index=['CustomerID', 'Sales Amount (k$)'],
                                                              columns='Attribute',
                                                              values='Value',
                                                              aggfunc='first').reset_index()

# Display the first few rows of the reshaped DataFrame
print(income_spending_data_pivot.head())

# Scatter plot for Sales vs. Annual Income
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Sales Amount (k$)', data=income_spending_data_pivot, hue='Gender', alpha=0.7)
plt.title("Sales Amount vs Annual Income by Gender")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Sales Amount (k$)")
plt.legend(title='Gender')
plt.show()

# Box plot to see spending patterns by gender
plt.figure(figsize=(8, 5))
sns.boxplot(x='Gender', y='Spending Score (1-100)', data=income_spending_data_pivot)
plt.title("Spending Score by Gender")
plt.xlabel("Gender")
plt.ylabel("Spending Score (1-100)")
plt.tight_layout()
plt.show()