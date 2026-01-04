import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
data = pd.read_csv("Housing.csv")

# Log transform target
data['log_price'] = np.log(data['price'])

# Binary encoding
cols = ["mainroad", "guestroom", "basement", 
        "hotwaterheating", "airconditioning", "prefarea"]

data[cols] = data[cols].replace({"no": 0, "yes": 1})

# Features & target
X = data.drop(["price", "log_price"], axis=1)
y = data["price"]

# One-hot encode furnishing status
X = pd.get_dummies(X, columns=["furnishingstatus"], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

# Train model
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Random Forest Results")
print("R2 Score :", r2)
print("MAE      :", mae)
print("RMSE     :", rmse)
