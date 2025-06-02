import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("FOOD DEMAND.csv")
print(df)

# Overview
print("Dataset Info:")
print(df.info())

# Check for duplicates
df = df.drop_duplicates()

# Check for missing values
print(df.isnull().sum())

print(df.describe())

print(df.isnull().sum())

# Distribution of checkout price
plt.figure(figsize=(8, 4))
sns.histplot(df['checkout_price'], bins=40, kde=True, color='blue')
plt.title('Checkout Price Distribution')
plt.xlabel('Checkout Price')
plt.ylabel('Count')
plt.tight_layout()
plt.legend()
plt.savefig("FOOD DEMAND.png")
plt.show()

# Promotions analysis
plt.figure(figsize=(6, 4))
sns.countplot(x='email_for_promotion', data=df)
plt.title("Email Promotions Distribution")
plt.tight_layout()
plt.legend()
plt.savefig("FOOD DEMAND.png")
plt.show()

# Average price by week
weekly_avg = df.groupby('week')['checkout_price'].mean().reset_index()
plt.figure(figsize=(8, 4))
sns.lineplot(data=weekly_avg, x='week', y='checkout_price')
plt.title("Average Checkout Price by Week")
plt.tight_layout()
plt.legend()
plt.savefig("FOOD DEMAND.png")
plt.show() 

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Choose one feature and the target variable
X = df[['checkout_price']]  # Feature: Checkout Price
y = df['base_price']  # Target: Number of Orders
# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict using the model
y_pred = model.predict(X)

# Print the coefficient and intercept
print(f"Coefficient (slope): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Plot the results
plt.figure(figsize=(8, 4))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Predicted Line')
plt.title("Checkout Price vs Number of Orders")
plt.xlabel("Checkout Price")
plt.ylabel("base_price")
plt.legend()
plt.tight_layout()
plt.show()

df.to_csv('cleaned FOOD DEMAND.csv',index= False)