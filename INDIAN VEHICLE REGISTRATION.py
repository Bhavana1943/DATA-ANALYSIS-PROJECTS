import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split  

df = pd.read_csv("Vehicle Registration In India.csv")
print(df)

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# Check for duplicates
df = df.drop_duplicates()

# Check for missing values
print(df.isnull().sum())


'''# Registrations by state
state_counts = df.groupby('state_name')['registrations'].sum().sort_values(ascending=False).head(10)
state_counts.plot(kind='barh', title='Top 100 States by Vehicle Registrations')
plt.legend()
plt.show()

# Registrations over time
monthly = df.groupby(df['date'].dt.to_period('M'))['registrations'].sum()
monthly.plot(title='Registrations over time')
plt.legend()
plt.show()

# Monthly registration trend
df['month'] = df['date'].dt.to_period('M')
monthly_reg = df.groupby('month')['registrations'].sum()
monthly_reg.plot(title='Monthly Vehicle Registrations', figsize=(10, 4), color='green')
plt.ylabel('Registrations')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# Transport vs Non-Transport
cat_reg = df.groupby('category')['registrations'].sum()
cat_reg.plot(kind='pie', autopct='%1.1f%%', title='Transport vs Non-Transport', ylabel='')
plt.legend()
plt.show()

# Registrations by vehicle type
type_reg = df.groupby('type')['registrations'].sum().sort_values(ascending=False).head(10)
type_reg.plot(kind='bar', title='Top 10 Vehicle Types by Registrations', color='coral', figsize=(10, 5))
plt.ylabel('Total Registrations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.show()


# Select only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Compute correlation matrix
correlation_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()'''

X = df[['state_code']]
y = df['registrations'] 

#2.Split the Data into Training and Testing Sets:
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#Step 4: Train a Linear Regression Model
#1.Create and Fit the Model:
model = LinearRegression()
model.fit(X_train, y_train)

#2.Print Model Coefficients
print(f'intercept:{model.intercept_}')
print(f'Coefficients:{model.coef_}')

y_pred = model.predict(X_test)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual registrations')
plt.ylabel('Predicted registrations')
plt.title('Actual vs Predicted Registrations')
plt.grid(True)
plt.tight_layout()
plt.show()

# Save cleaned version
df.to_csv('cleaned Indian Vehicle Report1.csv', index=False)
