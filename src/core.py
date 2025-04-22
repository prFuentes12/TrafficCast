import kagglehub
import os
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


# ============================
# DOWNLOAD & PREPOCESSING
# ============================

# Download the latest version of the dataset
path = kagglehub.dataset_download("rohith203/traffic-volume-dataset")

# Define the path to the 'data' folder outside of 'src'
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# Create the 'data' folder if it doesn't already exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Move the downloaded files to the 'data' folder
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    if os.path.isfile(file_path):
        shutil.move(file_path, os.path.join(data_dir, file))

train_df = pd.read_csv(data_dir+"/Train.csv")
test_df = pd.read_csv(data_dir+"/Test.csv")



# ============================
# EDA
# ============================

print("Train_df types:")
print(train_df.dtypes)

print("\nTest_df types")
print(test_df.dtypes)

# We will use the train_df for both training and validation
# because the test CSV is missing the target column ('traffic_volume') that we want to predict.
# Therefore, we will split the train_df into training and validation sets
# to train our model and evaluate its performance.


# Print the first 5 rows of the train dataset
print(train_df.head())


# Get the number of rows and columns in the train dataset
print("Train Dataset Shape (rows, columns):")
print(train_df.shape)


# Describe the training dataset to get a summary of the statist
print("Train Dataset Description:")
print(train_df.describe())

# Print the number of missing values for each column
print(train_df.isnull().sum())


# Get the unique values in the 'is_holiday' column
print(train_df['is_holiday'].unique())


# The 'is_holiday' column contains named holidays and many NaN values.
# These NaNs likely represent regular, non-holiday days.
# To avoid losing data and ensure the column is fully usable for analysis or modeling,
# we replace NaN with a new category: 'No Holiday'.
train_df['is_holiday'] = train_df['is_holiday'].fillna('No Holiday')

# After filling missing values in 'is_holiday', we check again to confirm there are no more nulls
# Check for missing values in train_df
print("Missing values in train_df after filling:")
print(train_df.isnull().sum())

# Get the unique values in the 'is_holiday' column after filling missing values
print(train_df['is_holiday'].unique())

# Get the unique values in the 'is_holiday' column after filling missing values
print(train_df['weather_type'].unique())

# Get the unique values in the 'is_holiday' column after filling missing values
print(train_df['weather_description'].unique())


# Convert to datetime format (if not already done)
train_df['date_time'] = pd.to_datetime(train_df['date_time'])

# Extract temporal features from 'date_time'
train_df['hour'] = train_df['date_time'].dt.hour                # Hour of the day (0–23)
train_df['dayofweek'] = train_df['date_time'].dt.dayofweek      # Day of the week (0=Monday, 6=Sunday)
train_df['month'] = train_df['date_time'].dt.month              # Month (1–12)
train_df['day'] = train_df['date_time'].dt.day                  # Day of the month
train_df['year'] = train_df['date_time'].dt.year                # Year

# Weekend indicator (1 if Saturday or Sunday, 0 otherwise)
train_df['is_weekend'] = train_df['dayofweek'].isin([5, 6]).astype(int)



# Apply One-Hot Encoding to the 'is_holiday' column
train_df = pd.get_dummies(train_df, columns=['is_holiday'], drop_first=True)

# Convert the newly created one-hot encoded columns from boolean to int
train_df[train_df.columns[train_df.columns.str.contains('is_holiday')]] = train_df[train_df.columns[train_df.columns.str.contains('is_holiday')]].astype(int)


# Print the result to verify the changes
print(train_df.head())


# Apply One-Hot Encoding to 'weather_type'
weather_type_dummies = pd.get_dummies(train_df['weather_type'], prefix='weather_type')

# Apply One-Hot Encoding to 'weather_description'
weather_desc_dummies = pd.get_dummies(train_df['weather_description'], prefix='weather_desc')

# Concatenate the encoded columns with the original dataframe
train_df = pd.concat([train_df, weather_type_dummies, weather_desc_dummies], axis=1)

# Convert the boolean columns to integers (0 or 1)
train_df[weather_type_dummies.columns] = train_df[weather_type_dummies.columns].astype(int)
train_df[weather_desc_dummies.columns] = train_df[weather_desc_dummies.columns].astype(int)



# Identify categorical columns (both original and after encoding)
categorical_columns = train_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# Identify non-categorical columns (numerical columns)
numerical_columns = train_df.select_dtypes(exclude=['object', 'category', 'bool']).columns.tolist()

# Print the categorical and non-categorical variables
print("Categorical variables:")
print(categorical_columns)

print("\nNon-Categorical variables:")
print(numerical_columns)




# Set visualization style and general figure size
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def plot_histograms(df, numerical_cols, dataset_name):
    """
    Plot histograms of numerical features.
    """
    df[numerical_cols].hist(bins=30, figsize=(15, 12), layout=(4, 3))
    plt.suptitle(f'Histograms of Numerical Features - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, numerical_cols, dataset_name):
    """
    Plot a heatmap of the correlation matrix for numerical features.
    """
    plt.figure(figsize=(12, 8))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix - {dataset_name}')
    plt.show()


def plot_categorical_distributions(df, categorical_cols, dataset_name):
    """
    Plot countplots for categorical variables.
    """
    for col in categorical_cols:
        plt.figure(figsize=(14, 6))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'{dataset_name} - Distribution of {col}')
        plt.xticks(rotation=45)
        plt.show()


def plot_boxplots_by_category(df, categorical_cols, dataset_name):
    """
    Plot boxplots of traffic volume by each categorical variable.
    """
    for col in categorical_cols:
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df, x=col, y='traffic_volume', order=df[col].value_counts().index)
        plt.title(f'{dataset_name} - Traffic Volume by {col}')
        plt.xticks(rotation=45)
        plt.show()


def plot_time_series(df, dataset_name):
    """
    Plot traffic volume over time using a lineplot.
    """
    if 'date_time' in df.columns:
        try:
            df = df.sort_values('date_time')
            plt.figure(figsize=(15, 5))
            sns.lineplot(data=df, x='date_time', y='traffic_volume')
            plt.title(f'{dataset_name} - Traffic Volume Over Time')
            plt.xticks(rotation=45)
            plt.show()
        except Exception as e:
            print(f"Error converting or plotting 'date_time' for {dataset_name}: {e}")


# Run EDA components individually on train_df (OPTIONAL)
# plot_histograms(train_df, numerical_cols, "Train Set")
# plot_correlation_heatmap(train_df, numerical_cols, "Train Set")
# plot_categorical_distributions(train_df, categorical_cols, "Train Set")
# plot_boxplots_by_category(train_df, categorical_cols, "Train Set")
# plot_time_series(train_df, "Train Set")



# ============================
# MODEL: FEATURE IMPORTANCE
# ============================

# Separate features (X) and target variable (y)

# Remove 'traffic_volume', 'date_time', and any other categorical variables from the features
X = train_df.drop(columns=['traffic_volume', 'date_time'] + categorical_columns)
y = train_df['traffic_volume']

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# ============================
# RANDOM FOREST MODEL
# ============================


# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get the feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame with features and their importance
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importances
})

# Sort by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Print the top 5 most important features
print("Top 5 Most Important Features:")
print(feature_importance_df.head(5))


# ============================
# TRAIN WITH THE TOP 5 MOST IMPORTANT FEATURES
# ============================

# Select the top 5 most important features
top_5_features = feature_importance_df.head(5)['feature'].values

# Create the new dataset with only the top 5 important features
X_train_top5 = X_train[top_5_features]
X_val_top5 = X_val[top_5_features]


# Train a Random Forest Regressor with these 5 features and regularization parameters to reduce overfitting
rf_model_top5 = RandomForestRegressor(
    n_estimators=100,          # Number of trees (you can adjust this too if needed)
    random_state=42,           # To ensure reproducibility
    max_depth=10,              # Limit tree depth to avoid overfitting
    min_samples_split=10,      # Require at least 10 samples to split a node
    min_samples_leaf=5         # Require at least 5 samples to be at a leaf node
)
rf_model_top5.fit(X_train_top5, y_train)

# Predictions on the training set
y_train_pred = rf_model.predict(X_train)

# Predictions on the validation set
y_val_pred = rf_model.predict(X_val)

# Calculate performance metrics on the training set
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = rf_model.score(X_train, y_train)  # R² score on training set

# Calculate performance metrics on the validation set
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = rf_model.score(X_val, y_val)  # R² score on validation set

# Print the results for both training and validation
print("Training Set Performance:")
print(f"Mean Squared Error (MSE): {train_mse}")
print(f"Root Mean Squared Error (RMSE): {train_rmse}")
print(f"Mean Absolute Error (MAE): {train_mae}")
print(f"R-squared (R²): {train_r2}")

print("\nValidation Set Performance:")
print(f"Mean Squared Error (MSE): {val_mse}")
print(f"Root Mean Squared Error (RMSE): {val_rmse}")
print(f"Mean Absolute Error (MAE): {val_mae}")
print(f"R-squared (R²): {val_r2}")