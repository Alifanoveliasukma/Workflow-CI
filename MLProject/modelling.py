import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Default")

X_train = pd.read_csv("housedataset_preprocessing/X_train.csv")
X_test = pd.read_csv("housedataset_preprocessing/X_test.csv")
y_train = pd.read_csv("housedataset_preprocessing/y_train.csv")
y_test = pd.read_csv("housedataset_preprocessing/y_test.csv")

mlflow.autolog()

with mlflow.start_run():

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2:", r2)
