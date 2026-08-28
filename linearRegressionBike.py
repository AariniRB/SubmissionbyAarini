import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
os.chdir(r"C:\Users\AARINI\Downloads")
train = pd.read_csv(r"C:\Users\AARINI\Downloads\train.csv")
train.drop_duplicates(keep='first', inplace=True)
train['datetime'] = pd.to_datetime(train['datetime'])
train['hour'] = train['datetime'].dt.hour
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['dayofweek'] = train['datetime'].dt.dayofweek
data = train.drop(columns=['datetime', 'casual', 'registered']).dropna().reset_index(drop=True)
numeric_cols = data.select_dtypes(include=[np.number])
correlation = numeric_cols.corr()
print("Correlation with total count:")
print(correlation['count'].abs().sort_values(ascending=False)[1:])
features_m1 = ['temp', 'atemp', 'humidity', 'windspeed', 'hour', 'year']
x1 = data[features_m1]
y1 = data[['count']]
Xtrain1, Xtest1, Ytrain1, Ytest1 = train_test_split(x1, y1, test_size=0.3, random_state=3)
# Variance Inflation Factor (VIF)
def calculateVIF(df):
    features = list(df.columns)
    model = LinearRegression()
    result = pd.DataFrame(index=['VIF'], columns=features)
    for target in features:
        predictors = [f for f in features if f != target]
        model.fit(df[predictors], df[target])
        r2 = model.score(df[predictors], df[target])
        result[target] = 1 / (1 - r2)
    return result

print("\nVIF Analysis:")
print(calculateVIF(Xtrain1).transpose())

# RMSE Helper Function
def rmse(test_y, predicted_y):
    rmse_test = np.sqrt(mean_squared_error(test_y, predicted_y))
    base_pred = np.repeat(np.mean(test_y), len(test_y))
    rmse_base = np.sqrt(mean_squared_error(test_y, base_pred))
    return {'RMSE-test from model': rmse_test, 'Base RMSE': rmse_base}

# OLS Fit — Model 1
X_train1_const = sm.add_constant(Xtrain1)
model_lin1 = sm.OLS(Ytrain1, X_train1_const).fit()
print(model_lin1.summary())

X_test1_const = sm.add_constant(Xtest1)
predictions_lin1_test = model_lin1.predict(X_test1_const)
print("\nModel 1 RMSE Results:")
print(rmse(Ytest1, predictions_lin1_test))

# Residuals & QQ-Plot for Model 1
predictions_lin1_train = model_lin1.predict(X_train1_const)
residuals1 = Ytrain1.iloc[:, 0] - predictions_lin1_train

sns.regplot(x=predictions_lin1_train, y=residuals1)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title('Residual plot - Model 1')
plt.show()

sm.qqplot(residuals1)
plt.title("Normal Q-Q Plot - Model 1")
plt.show()

# 3. DISTRIBUTION COMPARISON: RAW VS LOG
y2 = np.log1p(data[['count']]) # log1p handles log transformation safely

prices_df = pd.DataFrame({"1. Before": Ytrain1.iloc[:, 0], "2. Log Transformed": np.log1p(Ytrain1.iloc[:, 0])})
prices_df.hist()
plt.suptitle("Train data: count vs log(count + 1)")
plt.show()

# 4. MODEL 2 — log(count) ~ Continuous Features
Ytrain2_log, Ytest2_log = train_test_split(y2, test_size=0.3, random_state=3)

X_train2_const = sm.add_constant(Xtrain1)
model_lin2 = sm.OLS(Ytrain2_log, X_train2_const).fit()
print(model_lin2.summary())

X_test2_const = sm.add_constant(Xtest1)
predictions_lin2_test = model_lin2.predict(X_test2_const)

def rmse_log(test_y, predicted_y):
    t1 = np.expm1(test_y)
    t2 = np.expm1(predicted_y)
    rmse_test = np.sqrt(mean_squared_error(t1, t2))
    base_pred = np.repeat(np.mean(t1), len(t1))
    rmse_base = np.sqrt(mean_squared_error(t1, base_pred))
    return {'RMSE-test from model': rmse_test, 'Base RMSE': rmse_base}

print("\nModel 2 (Log Target) RMSE Results:")
print(rmse_log(Ytest2_log, predictions_lin2_test))

predictions_lin2_train = model_lin2.predict(X_train2_const)
residuals2 = Ytrain2_log.iloc[:, 0] - predictions_lin2_train

sns.regplot(x=predictions_lin2_train, y=residuals2)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title('Residual plot - Model 2')
plt.show()

sm.qqplot(residuals2)
plt.title("Normal Q-Q Plot - Model 2")
plt.show()

# 5. MODEL 3 — Full Model with One-Hot Encoded Categoricals
cat_cols = ['season', 'weather', 'year', 'month', 'dayofweek', 'hour']
x3 = pd.get_dummies(data.drop(columns=['count']), columns=cat_cols, drop_first=True, dtype=int)

Xtrain3, Xtest3 = train_test_split(x3, test_size=0.3, random_state=3)
Xtrain3 = Xtrain3.astype(float)

X_train3_const = sm.add_constant(Xtrain3)
model_lin3 = sm.OLS(Ytrain2_log, X_train3_const).fit()
print(model_lin3.summary())

X_test3_const = sm.add_constant(Xtest3)
X_test3_const = X_test3_const.reindex(columns=X_train3_const.columns, fill_value=0)
predictions_lin3_test = model_lin3.predict(X_test3_const)

print("\nModel 3 (Full Dummy Model) RMSE Results:")
print(rmse_log(Ytest2_log, predictions_lin3_test))

predictions_lin3_train = model_lin3.predict(X_train3_const)
residuals3 = Ytrain2_log.iloc[:, 0] - predictions_lin3_train

sns.regplot(x=predictions_lin3_train, y=residuals3)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title('Residual plot - Model 3')
plt.show()

sm.qqplot(residuals3)
plt.title("Normal Q-Q Plot - Model 3")
plt.show()
