from clearml import StorageManager

# Local file path
local_path = "../Data/Wildlife_Drone_Datasets.rar"

# Remote path in ClearML storage (s3://, gs://, azure://, or clearml://)
remote_path = "clearml://Wildlife/Data/Wildlife_Drone_Datasets.rar"

# Upload file
StorageManager.upload_file(local_path=local_path, remote_url=remote_path)

print("File uploaded to:", remote_path)