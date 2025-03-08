# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('weather_data.csv')

# Preprocess the dataset
# Handle missing values for numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Handle missing values for categorical columns
data['rain_or_not'].fillna(data['rain_or_not'].mode()[0], inplace=True)

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
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict the next day's rain
# Assuming the last row in the dataset is the most recent data
next_day_features = X.iloc[-1].values.reshape(1, -1)
next_day_prediction = model.predict(next_day_features)

# Output the prediction
if next_day_prediction[0] == 1:
    print("It will rain tomorrow.")
else:
    print("It will not rain tomorrow.")