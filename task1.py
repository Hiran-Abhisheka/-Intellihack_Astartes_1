# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('weather_data.csv')

# Preprocess the dataset
# Handle missing values for numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Handle missing values for categorical columns
data['rain_or_not'] = data['rain_or_not'].fillna(data['rain_or_not'].mode()[0])

# Encode 'rain_or_not' as binary
data['rain_or_not'] = data['rain_or_not'].apply(lambda x: 1 if x == 'Rain' else 0)

# Normalize/Standardize Features
features_to_scale = ['avg_temperature', 'humidity', 'avg_wind_speed']
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Define features and target
X = data.drop(['date', 'rain_or_not'], axis=1)
y = data['rain_or_not']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict the next 21 days' rain using a rolling window approach
# Calculate the rolling mean of the last 7 days for each numeric feature
rolling_window_size = 7
recent_data = data.iloc[-rolling_window_size:]
rolling_means = recent_data[numeric_cols].mean()

# Create future data based on rolling means
future_data = pd.DataFrame([rolling_means] * 21)

# Normalize/Standardize future features
future_data[features_to_scale] = scaler.transform(future_data[features_to_scale])

# Predict probabilities for the next 21 days
next_21_days_probabilities = model.predict_proba(future_data)[:, 1]

# Output the predictions
threshold = 0.5  # You can adjust this threshold based on your needs
for i, probability in enumerate(next_21_days_probabilities, start=1):
    if probability >= threshold:
        print(f"Day {i}: It will rain (Probability: {probability:.2f}).")
    else:
        print(f"Day {i}: It will not rain (Probability: {probability:.2f}).")