import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Read the cleaned data
data = pd.read_csv("D:\\Amogh_TY_Projects\\Semester 1\\MVA\\austin_weather.csv")

# Handle missing values
data = data.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN
data = data.dropna()  # Drop rows with NaN values

# Exclude non-numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
data = data[numeric_columns]

# Check dataset size
if len(data) < 2:
    raise ValueError("Insufficient data points for training and testing.")

# Define features (X) and target variable (Y)
X = data.drop(['PrecipitationSumInches'], axis=1)
Y = data['PrecipitationSumInches']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the neural network model
clf = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
clf.fit(X_train_scaled, Y_train)

# Evaluate the model on the test set
accuracy = clf.score(X_test_scaled, Y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Example input for prediction
inp = np.array([[74, 60, 45, 67, 49, 43, 33, 45, 57, 29.68, 10, 7, 2, 0, 20, 4, 31]])
inp_scaled = scaler.transform(inp)

# Print output
prediction = clf.predict(inp_scaled)
print('The precipitation in inches for the input is:', prediction)

# Plot the predicted vs actual precipitation on the test set
plt.scatter(Y_test, clf.predict(X_test_scaled))
plt.xlabel('Actual Precipitation (inches)')
plt.ylabel('Predicted Precipitation (inches)')
plt.title('Actual vs Predicted Precipitation')
plt.show()
