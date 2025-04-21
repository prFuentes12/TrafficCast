import kagglehub
import os
import shutil
import pandas as pd



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
