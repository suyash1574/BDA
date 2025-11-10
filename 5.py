import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import numpy as np


# Load data
data = pd.read_csv("Synthetic_Graduate_Admissions.csv")
X = data[['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research']]
y = data['Chance of Admit']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# Models
models = {
    "Linear": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor()
}

# Train & evaluate
for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    print(f"{name}: R²={r2_score(y_test,y_pred):.3f}, RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

# Predict for one student
sample = [[320,110,4,4.5,4,9,1]]
sample = scaler.transform(sample)
print("\nPredicted admission chance:")
for n,m in models.items():
    print(f"{n}: {m.predict(sample)[0]*100:.2f}%")