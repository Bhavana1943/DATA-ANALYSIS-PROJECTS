import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the Dataset
df = pd.read_csv("installed apps.csv")
print(df)

#check the data
print(df.shape)

df.drop_duplicates(inplace=True)
print(df)


#columns names
(print("column names:", df.columns))

#missing values
#print("missing values:", df.isnull().sum())

df.info()

'''#cleaning part
A_to_clean = ['Reviews', 'Installs', 'Price', 'Size']
for A in A_to_clean:
    if A in df.columns:
        df[A] = df[A].astype(str).str.replace('[+,\$]', '', regex=True)
        df[A] = df[A].str.replace('M', '', regex=True)
        df[A] = pd.to_numeric(df[A], errors='coerce')

# Drop duplicates and rows with missing key info
important = ['App', 'Category', 'Rating', 'Reviews', 'Installs']
df.drop_duplicates(inplace=True)
df.dropna(subset=[col for col in important if col in df.columns], inplace=True)        

# Drop rows with missing target values
df.dropna(subset=['Rating'], inplace=True)
df.dropna(subset=['Content Rating'], inplace=True)
df.dropna(subset=['Type'], inplace=True)
df.dropna(subset=['Current Ver'], inplace=True)
df.dropna(subset=['Android Ver'], inplace=True) 
df.dropna(subset=['Size'], inplace=True)     
print(df)

print("missing values:", df.isnull().sum())

#EDA
# Plot: Distribution of Ratings
sns.histplot(df['Rating'], bins=20, kde=True)
plt.title("Distribution of App Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.legend()
plt.show()

# Plot: Free vs Paid Apps
df['Type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Free vs Paid Apps')
plt.ylabel('')
plt.legend()
plt.show()

# Plot: Paid App Price Distribution
paid_apps = df[df['Price'] > 0]
paid_apps['Price'].plot(kind='hist', bins=30)
plt.title('Price Distribution of Paid Apps')
plt.xlabel('Price ($)')
plt.legend()
plt.show()

text = ' '.join(df['App'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of App Names')
plt.legend()
plt.show()

# Visualize the Relationships
#1.Sactter plots
sns.pairplot(df, x_vars=['Reviews', 'Installs', 'Price'], y_vars='Rating', height = 4, aspect=1,kind='scatter')
plt.show()

# Select only numerical columns for correlation analysis
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Compute the correlation matrix
correlation_matrix = numeric_df.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()



X = df[['Price','Reviews','Installs']]
y = df['Rating'] 

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

# Scatter plot: Actual vs Predicted Ratings
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.grid(True)
plt.tight_layout()
plt.show()

df.to_csv('cleaned Google Play Store App Analysis.csv',index= False)'''