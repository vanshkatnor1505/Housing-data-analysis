# 🏠 Housing Price Prediction (Machine Learning Project)
📌 Project Overview

This project focuses on predicting house prices using machine learning techniques.
Multiple regression models were implemented, evaluated, and compared to identify the most suitable approach based on data characteristics and performance.

📂 Dataset

Name: Housing.csv

Rows: 545

Columns: 13

Target Variable: price

Feature Types

Numerical: area, bedrooms, bathrooms, stories, parking, price

Binary (Yes/No): mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea

Categorical: furnishingstatus (furnished, semi-furnished, unfurnished)

🔍 Exploratory Data Analysis (EDA)
Key Observations

No missing values in the dataset

Price distribution is right-skewed

Strong positive correlation between:

price and area

price and bathrooms

Furnished houses tend to have higher prices

Houses with amenities (AC, prefarea, basement) are priced higher

Visualizations Performed

Price distribution (histogram + KDE)

Area vs Price scatter plot

Bedrooms vs Price

Furnishing status vs Price

Correlation heatmap (numerical features)

⚙️ Feature Engineering & Preprocessing
Target Transformation

Log transformation applied:

data['log_price'] = np.log(data['price'])


Used to:

 - Reduce skewness

 - Improve linear model performance

 - Encoding

Binary features encoded as:

yes → 1
no → 0


One-hot encoding for furnishingstatus:

    pd.get_dummies(drop_first=True)

Scaling

StandardScaler applied for linear models

Not used for tree-based models

🤖 Models Implemented
1️⃣ Linear Regression

Baseline model

Performance limited by multicollinearity

R² ≈ 0.67

2️⃣ Ridge Regression ✅ (Best Model)

Used to handle multicollinearity and stabilize coefficients.

Best alpha: 0.01 – 0.1

Performance:

R² ≈ 0.67

MAE ≈ 0.20 (log scale)

RMSE ≈ 0.25 (log scale)

📌 Chosen as final model

3️⃣ Lasso Regression

Performed feature selection

Slight underfitting observed

R² ≈ 0.65

Important Selected Features:

area

bathrooms

airconditioning

stories

prefarea

4️⃣ Random Forest Regression ❌

Tested to capture non-linear relationships.

Result:

R² dropped to ~0.61

Higher MAE & RMSE

Poor generalization

📉 Reason for failure:

Small dataset

Mostly linear relationships

Correlated features

Tree-based models overfit noise

📊 Model Comparison
Model	R² Score
Linear Regression	~0.67
Ridge Regression	~0.67 (Best)
Lasso Regression	~0.65
Random Forest	~0.61
🧠 Key Learnings

Model complexity does not guarantee better performance

Ridge regression is well-suited for small, linear datasets

Random Forests require larger datasets and non-linear patterns

Proper EDA and diagnostics are crucial for model selection

✅ Final Conclusion

Ridge Regression with log-transformed price is the most suitable model for this dataset due to its stability, interpretability, and superior generalization performance.

🚀 Next Steps

Residual diagnostics and assumption checks

Polynomial and interaction features

Apply workflow to a more complex dataset

🛠 Tech Stack

Python

Pandas, NumPy

Seaborn, Matplotlib

Scikit-learn

👏 Final Note

This project demonstrates end-to-end ML workflow:
EDA → Feature Engineering → Model Selection → Evaluation → Documentation
