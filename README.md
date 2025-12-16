Steel Industry Energy Consumption Regression



This project builds a robust regression pipeline to predict daily electricity usage (`Usage\_kWh`) in the steel industry using operational, environmental, and temporal features. It includes extensive preprocessing, feature engineering, model comparison, and interpretability visualizations.



&nbsp; Dataset Overview



* Source: Steel Industry Energy Consumption Dataset  
* Target Variable: `Usage\_kWh` (daily electricity usage in kilowatt-hours)
* Features:

1. Numeric: Reactive power metrics, power factors, CO₂ emissions, NSM, cyclical hour encodings, month, weekend flag
2. Categorical: WeekStatus, Day\_of\_week, Load Type (if present)



&nbsp; Preprocessing Pipeline



Numeric Features: Scaled using `RobustScaler` to handle outliers  

* Categorical Features: Encoded using `OneHotEncoder` with `handle\_unknown='ignore'`  
* ColumnTransformer: Combines both transformations into a unified pipeline 
* Time-Aware Split: Last 20% of data used as test set to preserve temporal integrity



Exploratory Data Analysis



Time-Series Insights

* Daily usage fluctuates significantly across the year, with seasonal and operational patterns.
* Hourly usage peaks during early production hours (08:00–10:00) and declines in the evening.
* Weekday usage is consistently higher than weekends, with Thursday showing peak demand.



Feature Relationships

* Strong positive correlation between `Usage\_kWh` and `CO2(tCO2)` and reactive power.
* Pairplot and heatmap reveal multicollinearity and outliers.
* Boxplots show reduced weekend usage and stable weekday patterns.



Modeling Pipeline



Models Implemented

* Linear Regression (baseline)
* Random Forest Regressor
* XGBoost Regressor



Pipeline Structure

```python

Pipeline(\[

&nbsp;   ('preprocessor', ColumnTransformer(...)),

&nbsp;   ('model', RandomForestRegressor(...))

])

