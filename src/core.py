import kagglehub
import os
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



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


# Set visualization style and general figure size
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

def eda_plots(df, dataset_name):
    """
    This function performs EDA on a given dataset (df) and prints various plots,
    including histograms, correlation heatmaps, countplots, and time series plots.
    The dataset is identified by dataset_name (e.g., "Train Set" or "Test Set").
    """
    
    # --- Histograms for Numerical Variables ---
    numerical_cols = [
        'air_pollution_index', 'humidity', 'wind_speed', 'wind_direction',
        'visibility_in_miles', 'dew_point', 'temperature',
        'rain_p_h', 'snow_p_h', 'clouds_all', 'traffic_volume'
    ]
    df[numerical_cols].hist(bins=30, figsize=(15, 12), layout=(4, 3))
    plt.suptitle(f'Histograms of Numerical Features - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # --- Correlation Heatmap ---
    plt.figure(figsize=(12, 8))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix - {dataset_name}')
    plt.show()
    
    # --- Countplots for Categorical Variables ---
    categorical_cols = ['is_holiday', 'weather_type', 'weather_description']
    for col in categorical_cols:
        plt.figure(figsize=(14, 6))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'{dataset_name} - Distribution of {col}')
        plt.xticks(rotation=45)
        plt.show()
        
    # --- Boxplots: Traffic Volume vs. Categorical Variables ---
    for col in categorical_cols:
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df, x=col, y='traffic_volume', order=df[col].value_counts().index)
        plt.title(f'{dataset_name} - Traffic Volume by {col}')
        plt.xticks(rotation=45)
        plt.show()
    
    # --- Time Series Plot for Traffic Volume ---
    if 'date_time' in df.columns:
        try:
            df['date_time'] = pd.to_datetime(df['date_time'])
            df = df.sort_values('date_time')
            plt.figure(figsize=(15, 5))
            sns.lineplot(data=df, x='date_time', y='traffic_volume')
            plt.title(f'{dataset_name} - Traffic Volume Over Time')
            plt.xticks(rotation=45)
            plt.show()
        except Exception as e:
            print(f"Error converting or plotting 'date_time' for {dataset_name}: {e}")

# Call the EDA function for the training set
eda_plots(train_df, "Train Set")




