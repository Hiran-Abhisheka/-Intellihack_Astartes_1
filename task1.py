# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('task1/weather_data.csv')

# Preprocess the dataset
# Handle missing values for numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Handle missing values for categorical columns
data['rain_or_not'].fillna(data['rain_or_not'].mode()[0], inplace=True)

# Encode 'rain_or_not' as binary
data['rain_or_not'] = data['rain_or_not'].apply(lambda x: 1 if x == 'Rain' else 0)

# Normalize/Standardize Features
features_to_scale = ['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure']
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Exploratory Data Analysis (EDA)
# Plot the distribution of average temperature
sns.histplot(data['avg_temperature'], kde=True)
plt.title('Distribution of Average Temperature')
plt.show()

# Plot the correlation matrix
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Train and Evaluate Machine Learning Models
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

# Optimize the Model
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Provide Probability of Rain
# Get probability predictions
y_prob = model.predict_proba(X_test)[:, 1]

# Display probabilities
print("Probability of rain:", y_prob)