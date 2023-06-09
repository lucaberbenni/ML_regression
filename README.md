# ML_regression
Predict count of rented bikes in a specific time.

1. **Data Import:**
   The code imports the training and test datasets from CSV files using pandas. The datasets contain information about bike rentals, including timestamps, weather conditions, and other relevant features.

2. **Feature Engineering:**
   The code extracts additional features from the timestamps, such as month, hour, and weekday, to capture temporal patterns in the data.

3. **Data Preparation:**
   The code separates the input features (X) from the target variable (y) in the training dataset. It also performs train-test split to create training and testing sets for model evaluation.

4. **Transformation Pipeline:**
   The code defines a transformation pipeline that applies specific transformations to different subsets of the input features. Categorical features are one-hot encoded, and numerical features are scaled using MinMaxScaler.

5. **Model Training and Hyperparameter Tuning:**
   The code creates a random forest regressor model and fits it to the transformed training data. It then performs hyperparameter tuning using GridSearchCV to find the best combination of hyperparameters (number of estimators and maximum depth) based on cross-validation.

6. **Model Evaluation:**
   The code evaluates the best estimator on the test data using the R-squared score, which measures the goodness of fit of the model. The R-squared score indicates the proportion of the variance in the target variable that is predictable from the input features.

7. **Prediction and Submission:**
   The code transforms the features in the test dataset using the transformation pipeline and makes predictions using the best estimator. The predictions are saved in a DataFrame, along with the corresponding timestamps, and then saved as a CSV file for submission.
