import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

# load csv data
data = pd.read_csv("Housing.csv")

# feature engeneering 
# for log transformation
data['log_price'] = np.log(data['price'])   

cols = ["mainroad", "guestroom", "basement", 
        "hotwaterheating", "airconditioning" , "prefarea"]

data[cols] = data[cols].replace({"no": 0 , "yes": 1})


# Seprate features and target
X = data.drop(["price", "log_price"], axis=1)
Y = data["log_price"]

# one_hot encoding
X = pd.get_dummies(X , columns=['furnishingstatus'], drop_first=True)

# test-train split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train linear regression 
model = LinearRegression()
model.fit(X_train, Y_train)

#predictions
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# calculate r2 score
r2 = r2_score(Y_test, Y_test_pred)
mae = mean_absolute_error(Y_test, Y_test_pred)
mse = mean_squared_error(Y_test, Y_test_pred)
rmse = np.sqrt(mse)

# print values 
print("R2 score : " , r2)
print("MAE:", mae)
print("RMSE:", rmse)