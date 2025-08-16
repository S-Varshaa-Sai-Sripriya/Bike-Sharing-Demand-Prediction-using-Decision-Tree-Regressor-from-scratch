# 🚲 Bike Sharing Demand Prediction using Decision Tree Regressor from scratch

This project implements a Decision Tree Regressor from scratch and applies it to the popular Bike Sharing Dataset. The goal is to predict the number of bikes rented based on environmental and seasonal features.

---

## Datset Explanation:

### Bike Sharing Demand Dataset
This dataset contains historical bike rental counts along with environmental and seasonal information.

- **Number of Instances**: ~17,379 (hourly data)  
- **Number of Features**: 12 + 1 target  
- **Target Variable**: `count` (total number of bike rentals)

**Feature Details**:

- `datetime` – Hourly date and time of record.  
- `season` – Season (1: Winter, 2: Spring, 3: Summer, 4: Fall).  
- `holiday` – Whether the day is a holiday (0: No, 1: Yes).  
- `workingday` – Whether the day is a working day (0: No, 1: Yes).  
- `weather` – Weather situation (1–4 scale).  
- `temp` – Temperature in Celsius.  
- `atemp` – “Feels like” temperature.  
- `humidity` – Relative humidity.  
- `windspeed` – Wind speed.  
- `casual` – Count of casual users (non-registered).  
- `registered` – Count of registered users.  
- `count` – Total number of rentals (casual + registered).

This dataset is useful for:

- Regression tasks to predict bike rental demand.  
- Time-series analysis and feature engineering for real-world applications.  

---

## Results

Mean Squared Error (MSE): 11325.0901
R² Score:                0.6424

---

## Insights:

The model explains 64% of the variance in bike rental demand.

Predictions are off by about 106 bikes on average.

---

## Future Improvements

    Hyperparameter tuning (e.g., max_depth, min_samples_split)

    Feature engineering (e.g., seasonal effects, interactions)

    Ensemble models: Random Forest, Gradient Boosting

    Cross-validation for more robust evaluation
