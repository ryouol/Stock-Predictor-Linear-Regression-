Data Acquisition and Preparation:
The program uses the yfinance library to fetch historical closing stock prices for a specified ticker symbol (like AAPL for Apple Inc.) between given start and end dates. The data includes the stock's closing price for each trading day, which is then enhanced by adding a 'Day' column that represents each date as the day of the year.
Model Training and Evaluation:
The stock data is split into features (Day of the year) and target (Close price). This dataset is further divided into training and testing sets. A linear regression model is then trained on the training set, which learns to predict the closing price based on the day of the year. The model's effectiveness is evaluated using the test set, and the accuracy is calculated to understand how well the model performs in predicting stock prices.
This model has a 9.96% accuracy ![Figure_3](https://github.com/ryouol/Stock-Predictor-Linear-Regression-/assets/125412884/24b82916-3025-48c5-962e-34a00c753067)