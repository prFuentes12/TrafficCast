import kagglehub

# Download latest version
path = kagglehub.dataset_download("rohith203/traffic-volume-dataset")

print("Path to dataset files:", path)