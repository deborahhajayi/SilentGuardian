import kagglehub

# Download the fall dataset into your repo
path = kagglehub.dataset_download("tuyenldvn/falldataset-imvia", path="data/raw/")

print("Path to dataset files:", path)
