import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.chdir(r"C:\Users\AARINI\Downloads")


# 1. Loading & Understanding the Dataset

bike = pd.read_csv(r"C:\Users\AARINI\Downloads\train.csv")

print("First 5 rows:")
print(bike.head())

print("\nLast 5 rows:")
print(bike.tail())

print("\nNumber of rows and columns:", bike.shape)
print("\nColumn names:", list(bike.columns))
print("\nData types of each column:")
print(bike.dtypes)
print("\nBasic statistics (numerical columns):")
print(bike.describe())

# Understanding features: numerical vs categorical
# season, holiday, workingday and weather are stored as integers but
# represent categories/labels, not continuous quantities, so they are
# treated as categorical here even though pandas reads them as int64.
categorical_cols = ['season', 'holiday', 'workingday', 'weather']
numerical_cols = [c for c in bike.columns if c not in categorical_cols + ['datetime']]
print("\nNumerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

print("\nUnique value counts for categorical columns:")
for col in categorical_cols:
    print(f"{col}: {bike[col].nunique()} unique values -> {sorted(bike[col].unique())}")
    print(bike[col].value_counts().sort_index())

print("\nDate range covered by the data:")
print(pd.to_datetime(bike['datetime']).min(), "to", pd.to_datetime(bike['datetime']).max())

# Target variable
target = 'count'
print(f"\nTarget variable selected: '{target}'")
print("Reasoning: 'count' is the total number of bikes rented in a given hour")
print("(count = casual + registered). It is a continuous numeric variable that")
print("directly measures the quantity a bike-sharing operator cares about most")
print("(hourly demand), making this a regression problem. 'casual' and")
print("'registered' are excluded from the feature set later since they sum")
print("exactly to 'count' and would leak the answer into the model.")

print("\nInitial Observations:")
print("1. 'datetime' is stored as a string/object column, so it must be parsed")
print("   into an actual datetime type and broken into hour/day/month/year")
print("   features before it can be used numerically by any model.")
print("2. 'count' ranges from 1 to 977 with a mean around 192 and a std of 181")
print("   -- the distribution is right-skewed, i.e. most hours have modest")
print("   demand while a smaller number of hours see very high demand.")
print("3. 'weather' is heavily imbalanced: category 1 (clear) makes up the")
print("   large majority of rows, while category 4 (heavy rain/snow) appears")
print("   only once in the whole dataset -- an extreme rare-category case.")


# 2. Data Cleaning

print("\nMissing values per column:")
print(bike.isnull().sum())
print("No missing values were found in any column, so no imputation is required.")

duplicate_count = bike.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")
bike.drop_duplicates(keep='first', inplace=True)
bike.reset_index(drop=True, inplace=True)
print("drop_duplicates() was still run defensively in case duplicates appear")
print("if the dataset is refreshed or re-downloaded later.")

print("\nChecking for invalid/inconsistent values:")
invalid_humidity = (bike['humidity'] == 0).sum()
print(f"Rows with humidity == 0: {invalid_humidity}")
print("0% humidity is not physically realistic for an outdoor weather sensor")
print("reading, so this is treated as an invalid/missing-value placeholder")
print("and imputed with the median humidity (median is used instead of mean")
print("since humidity can be skewed by weather extremes).")
median_humidity = bike.loc[bike['humidity'] != 0, 'humidity'].median()
bike.loc[bike['humidity'] == 0, 'humidity'] = median_humidity

invalid_windspeed = (bike['windspeed'] == 0).sum()
print(f"\nRows with windspeed == 0: {invalid_windspeed}")
print("Unlike humidity, 0 windspeed is physically plausible (a genuinely calm")
print("hour), so these rows are kept as-is rather than treated as invalid.")

negative_values = (bike[['temp', 'atemp', 'humidity', 'windspeed',
                          'count', 'casual', 'registered']] < 0).sum().sum()
print(f"\nNegative values found across temp/atemp/humidity/windspeed/count/")
print(f"casual/registered: {negative_values} -- none found, no correction needed.")


# Feature engineering: break datetime into usable numeric parts
bike['datetime'] = pd.to_datetime(bike['datetime'])
bike['hour'] = bike['datetime'].dt.hour
bike['day'] = bike['datetime'].dt.day
bike['month'] = bike['datetime'].dt.month
bike['year'] = bike['datetime'].dt.year
bike['dayofweek'] = bike['datetime'].dt.dayofweek


# 3. Data Visualization

# Distribution plot
plt.figure(figsize=(8, 5))
sns.histplot(bike['count'], bins=50, kde=True)
plt.title("Distribution of Total Hourly Bike Rentals (count)")
plt.xlabel("Count of rentals")
plt.ylabel("Frequency")
plt.show()
print("Observation: 'count' is heavily right-skewed -- most hours see fewer")
print("than 300 rentals, while a small number of hours spike above 700. This")
print("suggests a log transform of the target could help models (like linear")
print("regression) that assume roughly normal, symmetric residuals.")

# Relationship plot between two variables
plt.figure(figsize=(8, 5))
sns.scatterplot(x='temp', y='count', data=bike, alpha=0.3)
plt.title("Relationship Between Temperature and Rental Count")
plt.xlabel("Temperature (deg C)")
plt.ylabel("Count of rentals")
plt.show()
print("Observation: rentals generally rise as temperature increases, but the")
print("relationship flattens/declines at very high temperatures -- a mild")
print("non-linear pattern rather than a straight line.")

# Target-variable plot
plt.figure(figsize=(9, 5))
sns.boxplot(x='hour', y='count', data=bike)
plt.title("Rental Count by Hour of Day (Target Variable)")
plt.xlabel("Hour of day")
plt.ylabel("Count of rentals")
plt.show()
print("Observation: demand is clearly cyclical, peaking sharply around 8am")
print("and 5-6pm (commute hours) with a dip overnight -- hour of day is")
print("likely to be one of the strongest predictors of count.")


# 4. Data Preprocessing

# Features & target
X = bike.drop(columns=['count', 'casual', 'registered', 'datetime'])
y = bike['count']

# Categorical encoding (one-hot, dropping first level to avoid redundancy)
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print("\nFeatures after one-hot encoding categorical columns:")
print(X.columns.tolist())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=3
)
print("\nTrain/test split shapes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# Feature scaling (fit on train only, then applied to test, to avoid leakage)
scale_cols = ['temp', 'atemp', 'humidity', 'windspeed']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

print("\nFeature scaling applied to continuous columns:", scale_cols)
print("StandardScaler was fit only on X_train and then used to transform")
print("X_test, which prevents information from the test set leaking into")
print("the scaling parameters. This matters most for distance-based models")
print("like KNN, where unscaled features with larger numeric ranges would")
print("dominate the distance calculation.")
print("\nScaled X_train preview:")
print(X_train_scaled[scale_cols].head())

print("\nData is now ready in X_train, X_test, y_train, y_test")
print("(and X_train_scaled/X_test_scaled for scale-sensitive models like KNN)")
print("for the 5 ML models to be built next.")