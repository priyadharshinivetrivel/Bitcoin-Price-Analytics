import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from xgboost import plot_importance

# ==========================
# LOAD DATASET
# ==========================

print("Loading Dataset...")

df = pd.read_csv(
    "Bitcoin_1_1_2020-1_1_2025_historical_data_coinmarketcap.csv",
    sep=";"
)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==========================
# DATA CLEANING
# ==========================

df['timeOpen'] = pd.to_datetime(df['timeOpen'])

df = df.sort_values('timeOpen')

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================
# EDA
# ==========================

plt.figure(figsize=(12,6))
plt.plot(df['timeOpen'], df['close'])
plt.title("Bitcoin Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df['timeOpen'], df['volume'])
plt.title("Bitcoin Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.show()

numeric_df = df.select_dtypes(include='number')

plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=False)
plt.title("Correlation Matrix")
plt.show()

# ==========================
# FEATURE ENGINEERING
# ==========================

df['Lag1'] = df['close'].shift(1)
df['Lag2'] = df['close'].shift(2)
df['Lag3'] = df['close'].shift(3)
df['Lag7'] = df['close'].shift(7)

df['MA7'] = df['close'].rolling(7).mean()
df['MA30'] = df['close'].rolling(30).mean()

df['Volatility'] = df['close'].rolling(7).std()

df.dropna(inplace=True)

# ==========================
# FEATURES & TARGET
# ==========================

X = df[
    [
        'open',
        'high',
        'low',
        'volume',
        'marketCap',
        'circulatingSupply',
        'Lag1',
        'Lag2',
        'Lag3',
        'Lag7',
        'MA7',
        'MA30',
        'Volatility'
    ]
]

y = df['close']

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================
# SVR MODEL
# ==========================

print("\nTraining SVR...")

svr = SVR()

svr.fit(X_train, y_train)

svr_pred = svr.predict(X_test)

# ==========================
# XGBOOST MODEL
# ==========================

print("Training XGBoost...")

xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

# ==========================
# EVALUATION
# ==========================

print("\n========== SVR RESULTS ==========")

print("MAE :", mean_absolute_error(y_test, svr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, svr_pred)))
print("R2  :", r2_score(y_test, svr_pred))

print("\n========== XGBOOST RESULTS ==========")

print("MAE :", mean_absolute_error(y_test, xgb_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, xgb_pred)))
print("R2  :", r2_score(y_test, xgb_pred))

# ==========================
# ACTUAL VS PREDICTED
# ==========================

plt.figure(figsize=(12,6))

plt.plot(
    y_test.values,
    label="Actual Price"
)

plt.plot(
    xgb_pred,
    label="Predicted Price"
)

plt.title("Actual vs Predicted Bitcoin Price")

plt.xlabel("Days")

plt.ylabel("Price")

plt.legend()

plt.show()

# ==========================
# FEATURE IMPORTANCE
# ==========================

plt.figure(figsize=(10,8))

plot_importance(xgb)

plt.title("Feature Importance")

plt.show()

# ==========================
# NEXT 7 DAY FORECAST
# ==========================

print("\nNext 7 Day Forecast")

last_row = X.iloc[-1:].copy()

future_predictions = []

for i in range(7):

    pred = xgb.predict(last_row)[0]

    future_predictions.append(pred)

for day, value in enumerate(
    future_predictions,
    start=1
):

    print(
        f"Day {day}: {value:.2f}"
    )

# ==========================
# SAVE MODELS
# ==========================

joblib.dump(
    xgb,
    "xgboost_model.pkl"
)

joblib.dump(
    svr,
    "svr_model.pkl"
)

print("\nModels Saved Successfully!")

print("\nProject Completed Successfully!")