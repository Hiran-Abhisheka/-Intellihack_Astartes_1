# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# 1. DATA LOADING AND PREPROCESSING
print("1. DATA LOADING AND PREPROCESSING")
print("--------------------------------")

# Load the dataset
data = pd.read_csv('weather_data.csv')

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("\nFirst few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Convert date column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Extract additional features from date
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_week'] = data['date'].dt.dayofweek

# Handle missing values
# For numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].mean())

# For categorical columns (specifically rain_or_not)
data['rain_or_not'] = data['rain_or_not'].fillna(data['rain_or_not'].mode()[0])

# Encode 'rain_or_not' as binary (1 for Rain, 0 for No Rain)
data['rain_or_not'] = data['rain_or_not'].apply(lambda x: 1 if x == 'Rain' else 0)

print("\nAfter preprocessing, missing values in each column:")
print(data.isnull().sum())

# 2. EXPLORATORY DATA ANALYSIS (EDA)
print("\n2. EXPLORATORY DATA ANALYSIS")
print("---------------------------")

# Set up the figure size for all plots
plt.figure(figsize=(15, 10))

# Distribution of rain/no rain days
plt.subplot(2, 3, 1)
sns.countplot(x='rain_or_not', data=data)
plt.title('Distribution of Rain vs No Rain Days')
plt.xlabel('Rain (1) / No Rain (0)')
plt.ylabel('Count')

# Distribution of numeric features
plt.subplot(2, 3, 2)
sns.histplot(data['avg_temperature'], kde=True)
plt.title('Distribution of Average Temperature')

plt.subplot(2, 3, 3)
sns.histplot(data['humidity'], kde=True)
plt.title('Distribution of Humidity')

plt.subplot(2, 3, 4)
sns.histplot(data['avg_wind_speed'], kde=True)
plt.title('Distribution of Average Wind Speed')

plt.subplot(2, 3, 5)
sns.histplot(data['cloud_cover'], kde=True)
plt.title('Distribution of Cloud Cover')

plt.subplot(2, 3, 6)
sns.histplot(data['pressure'], kde=True)
plt.title('Distribution of Pressure')

plt.tight_layout()
plt.savefig('distributions.png')
plt.show()

# Relationship between features and target
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.boxplot(x='rain_or_not', y='avg_temperature', data=data)
plt.title('Temperature vs Rain')

plt.subplot(2, 3, 2)
sns.boxplot(x='rain_or_not', y='humidity', data=data)
plt.title('Humidity vs Rain')

plt.subplot(2, 3, 3)
sns.boxplot(x='rain_or_not', y='avg_wind_speed', data=data)
plt.title('Wind Speed vs Rain')

plt.subplot(2, 3, 4)
sns.boxplot(x='rain_or_not', y='cloud_cover', data=data)
plt.title('Cloud Cover vs Rain')

plt.subplot(2, 3, 5)
sns.boxplot(x='rain_or_not', y='pressure', data=data)
plt.title('Pressure vs Rain')

plt.tight_layout()
plt.savefig('feature_vs_target.png')
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation = data.drop('date', axis=1).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation.png')
plt.show()

# 3. DATA PREPARATION FOR MODELING
print("\n3. DATA PREPARATION FOR MODELING")
print("------------------------------")

# Define features and target
features = ['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure', 
            'month', 'day', 'day_of_week']
X = data[features]
y = data['rain_or_not']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Feature names: {features}")

# 4. MODEL TRAINING AND EVALUATION
print("\n4. MODEL TRAINING AND EVALUATION")
print("------------------------------")

# Define models to evaluate
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Initialize variables to track best model
best_model_name = None
best_score = 0

# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Update best model if necessary
    if accuracy > best_score:
        best_score = accuracy
        best_model_name = name
    
    # Print evaluation metrics
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
    plt.show()
    
    # Plot ROC curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_{name.replace(" ", "_").lower()}.png')
        plt.show()

print(f"\nBest Model: {best_model_name} with accuracy {best_score:.4f}")

# 5. HYPERPARAMETER TUNING
print("\n5. HYPERPARAMETER TUNING")
print("----------------------")

# Based on typical good performance, let's tune the Gradient Boosting model
# You could also tune the best model from the previous step
model_to_tune = GradientBoostingClassifier(random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(
    estimator=model_to_tune,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

print("Starting grid search...")
grid_search.fit(X_train_scaled, y_train)

# Get best parameters
print("\nBest Parameters:")
print(grid_search.best_params_)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, y_pred)
print(f"\nBest Model Accuracy: {best_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. FEATURE IMPORTANCE
print("\n6. FEATURE IMPORTANCE")
print("------------------")

# Extract feature importance
feature_importance = best_model.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance (Gradient Boosting)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# 7. PREDICTING NEXT 21 DAYS
print("\n7. PREDICTING NEXT 21 DAYS")
print("------------------------")

# Calculate rolling means of the last 7 days for numeric features
rolling_window_size = 7
recent_data = data.iloc[-rolling_window_size:]

# Create a DataFrame to store predictions
future_predictions = pd.DataFrame()

# Generate dates for the next 21 days
last_date = data['date'].max()
future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(21)]
future_predictions['date'] = future_dates

# Extract month, day, and day_of_week from future dates
future_predictions['month'] = future_predictions['date'].dt.month
future_predictions['day'] = future_predictions['date'].dt.day
future_predictions['day_of_week'] = future_predictions['date'].dt.dayofweek

# Generate features for the next 21 days based on recent trends
# For simplicity, use the mean of the last 7 days for numeric features
for feature in ['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure']:
    future_predictions[feature] = recent_data[feature].mean()

# Scale the features
X_future = future_predictions[features]
X_future_scaled = scaler.transform(X_future)

# Predict probabilities for the next 21 days
probabilities = best_model.predict_proba(X_future_scaled)[:, 1]
future_predictions['rain_probability'] = probabilities

# Add binary predictions based on a threshold of 0.5
future_predictions['rain_prediction'] = (probabilities >= 0.5).astype(int)

# Display the predictions
print("\nPredictions for the next 21 days:")
print(future_predictions[['date', 'rain_probability', 'rain_prediction']].to_string(index=False))

# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.bar(range(21), probabilities, color=['skyblue' if p < 0.5 else 'navy' for p in probabilities])
plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.xticks(range(21), [f"Day {i+1}" for i in range(21)], rotation=45)
plt.ylabel('Probability of Rain')
plt.title('Rain Probability Forecast for the Next 21 Days')
plt.legend()
plt.tight_layout()
plt.savefig('rain_forecast.png')
plt.show()

# Save predictions to CSV
future_predictions.to_csv('rain_forecast_21_days.csv', index=False)
print("\nForecast saved to 'rain_forecast_21_days.csv'")

# 8. SUMMARY AND CONCLUSION
print("\n8. SUMMARY AND CONCLUSION")
print("-----------------------")
print("We've built a machine learning model to predict rain probability for the next 21 days.")
print(f"The final model achieved an accuracy of {best_accuracy:.4f} on the test data.")
print("The most important features for prediction were:", ", ".join(importance_df['Feature'].head(3).tolist()))
print("The forecast for the next 21 days has been generated and saved to 'rain_forecast_21_days.csv'.")