gitimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import chardet

# Here I have written  the code of Detecting file encoding
file_path = r"C:\Users\paras\Downloads\GrowthLink Internship task\car_purchasing.csv"
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read(100000))  
    encoding = result['encoding']

#Here I have written   Loading  dataset with correct encoding
df = pd.read_csv(file_path, encoding=encoding)

# Here I have written the code of Displaying basic info
print("Dataset Info:")
print(df.info())

# Here I have written the code of Checking for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Here I have written the code of Filling missing values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Here I have written the code of Detecting and removing outliers using Z-score method
from scipy.stats import zscore
df = df[(np.abs(zscore(df.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

# Drop non-relevant columns (customer name, email, country) if they exist
columns_to_drop = ["customer name", "customer e-mail", "country"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Defined features and target variable
target_column = "car purchase amount"
if target_column not in df.columns:
    print(f"Error: Column '{target_column}' not found in dataset.")
    exit()

X = df.drop(columns=[target_column])
y = df[target_column]

# Here I have written the code of Spliting dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Here I have written the code of Scaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Here I have written the code of Training Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Here I have written the code of Making predictions
y_pred = model.predict(X_test)

# Here I have written the code of Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Here I have written the code of Ploting Actual vs Predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
