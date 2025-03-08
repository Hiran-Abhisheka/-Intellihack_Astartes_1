# Intellihack_Astartes_1

# Weather Forecasting for Smart Agriculture 🌧️🌱

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

This project develops a machine learning model for predicting hyperlocal rain probability for agricultural planning. Traditional weather forecasts often lack the local precision needed for effective farm management. Our model analyzes historical weather data to provide accurate 21-day rain forecasts to help farmers optimize irrigation, planting, and harvesting schedules.

## 📝 Summary

This project applies machine learning techniques to predict daily rain probability based on local weather conditions. Using ensemble methods with feature engineering and hyperparameter optimization, we created a robust prediction model that:

- Preprocesses weather data to handle missing and inconsistent values
- Extracts meaningful temporal features from dates
- Evaluates multiple ML algorithms to find the most accurate prediction model
- Generates 21-day forecasts with probability scores for rain likelihood
- Visualizes predictions for easy interpretation by farmers

The resulting system achieves high accuracy in predicting local rainfall, enabling farmers to make data-driven decisions about agricultural operations. The model considers various meteorological indicators including temperature, humidity, wind speed, cloud cover, and atmospheric pressure to deliver reliable predictions.

## 📑 Table of Contents

- [Project Overview](#overview)
- [Summary](#summary)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [System Design](#system-design)
- [Visualizations](#visualizations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)

## 📊 Dataset Description

The analysis uses `weather_data.csv`, containing daily weather observations for 300 days:

| Feature | Description |
|---------|-------------|
| `date` | Date of observation |
| `avg_temperature` | Average temperature in °C |
| `humidity` | Humidity percentage |
| `avg_wind_speed` | Average wind speed in km/h |
| `rain_or_not` | Binary label (Rain/No Rain) |
| `cloud_cover` | Cloud cover percentage |
| `pressure` | Atmospheric pressure in hPa |

## 🔧 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/weather-forecasting.git
   cd weather-forecasting
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, install packages individually:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

## 🚀 Usage

1. Ensure the dataset `weather_data.csv` is in the project directory

2. Run the main script:
   ```bash
   python weather_forecasting.py
   ```

3. The script will:
   - Preprocess weather data
   - Perform exploratory data analysis
   - Train multiple prediction models
   - Generate 21-day rain forecasts
   - Create visualization of predictions
   - Save the forecast to `rain_forecast_21_days.csv`

## 🔬 Methodology

The analysis follows these steps:

1. **Data Preprocessing**
   - Missing value imputation for numeric and categorical features
   - Date feature extraction (month, day, day_of_week)
   - Binary encoding of rain labels

2. **Exploratory Data Analysis**
   - Distribution analysis of meteorological features
   - Correlation analysis between weather variables
   - Feature relationship with rain occurrence

3. **Model Development**
   - Training multiple algorithms (Logistic Regression, Decision Trees, Random Forests, Gradient Boosting)
   - Cross-validation for robust performance evaluation
   - Hyperparameter tuning using GridSearchCV

4. **Feature Engineering**
   - Extraction of temporal patterns
   - Feature importance analysis
   - Standardization of numeric features

5. **Forecasting**
   - Rolling window approach for generating future features
   - Probability calculation for rain prediction
   - Threshold-based binary classification

## 📈 Results

The analysis produces:

- **Trained Model**: Optimized machine learning model (typically Gradient Boosting) for rain prediction
- **Feature Importance**: Identification of the most influential weather variables for rain prediction
- **21-Day Forecast**: Daily probability of rain for the next 21 days
- **Accuracy Metrics**: Performance evaluation of the model using standard classification metrics

## 🖥️ System Design

Our MLOps pipeline includes:

1. **Data Ingestion**: Collection of real-time weather data from IoT sensors at 1-minute intervals
2. **Data Processing**: Robust preprocessing to handle sensor malfunctions and missing data
3. **Model Inference**: Daily prediction of rain probability using the trained model
4. **Delivery Interface**: User-friendly visualization of forecasts for farmers
5. **Monitoring**: Continuous evaluation of prediction accuracy and model retraining

## 📊 Visualizations

The project produces multiple visualizations:

- Distribution of weather variables and their relationship with rainfall
- Feature importance charts for model interpretability
- Correlation heatmaps between weather variables
- 21-day rain probability forecast chart
- ROC curves and confusion matrices for model evaluation

## 🔮 Future Improvements

- Integration with satellite imagery data for enhanced prediction
- Implementation of deep learning models for capturing complex weather patterns
- Development of a mobile application for farmers to access predictions
- Addition of crop-specific recommendations based on weather forecasts
- Implementation of anomaly detection for extreme weather events

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
