This project focuses on predicting bike sharing demand using machine learning techniques. The goal is to develop a model that can accurately estimate the number of bike rentals based on various factors such as weather conditions, time of day, and other relevant features.

### Dataset

The dataset used for this project consists of historical bike sharing data, which includes information about the number of bikes rented, weather conditions, and date/time. It is split into a training dataset and a test dataset.

### Feature Engineering

To improve the model's performance, several preprocessing steps were applied to the dataset. The date/time information was extracted and transformed into separate columns for month, hour, and weekday. This allows the model to capture any potential patterns or trends related to these temporal factors. Categorical variables were one-hot encoded, and numerical variables were scaled using a MinMaxScaler.

### Model Training

A Random Forest Regressor was chosen as the predictive model due to its ability to handle both numerical and categorical features effectively. A transformation pipeline was created to preprocess the data and feed it into the model. The model was then trained on the transformed training dataset.

### Hyperparameter Optimization

To improve the model's performance further, a grid search was performed to find the best combination of hyperparameters for the Random Forest Regressor. The grid search was conducted using cross-validation and evaluated based on the coefficient of determination (R2 score). The best estimator from the grid search was selected and used for prediction.

### Prediction and Submission

The trained model was applied to the test dataset to make predictions on bike sharing demand. The R2 score was calculated to evaluate the model's performance on the test dataset. Finally, the model was used to predict bike sharing demand for a separate dataset (Kaggle competition) and the results were saved in a submission file for evaluation.

This project demonstrates the application of machine learning techniques for predicting bike sharing demand. The code provided can be used as a reference for similar prediction tasks or as a starting point for further exploration and improvement.

Feel free to explore the code and dataset provided to gain insights into the bike sharing demand prediction task. If you have any questions or suggestions, please feel free to reach out.

Happy coding!
