from clearml import Task
import os

# Create a dataset experiment
task = Task.init(project_name="WildlifeDetector", task_name="Upload dataset")

task.set_packages([
    "boto3==1.37.38",
    "keras==3.3.3",       # good match with TF 2.20.0
    "matplotlib==3.10.1",
    "numpy==2.1.1",
    "pandas==2.2.3",
    "pillow==11.2.1",
    "scikit_learn==1.6.1",
    "seaborn==0.13.2",
    "tensorflow==2.20.0",
    "clearml==1.18.0",
])

# Only create the task, we will actually execute it later
task.execute_remotely()

# Path to the zipped dataset file
zipped_dataset = "../../Data/Wildlife_Drone_Datasets.rar"

print(f"Current working directory: {os.getcwd()}")

# Resolve and verify the file path
absolute_path = os.path.abspath(zipped_dataset)
print(f"Resolved absolute path: {absolute_path}")
if not os.path.exists(absolute_path):
    raise FileNotFoundError(f"File '{absolute_path}' does not exist!")

# Upload the zipped file as an artifact
task.upload_artifact(name="dataset", artifact_object=zipped_dataset)

print(f"Uploaded zipped dataset '{zipped_dataset}' as an artifact.")

# We are done
print("Done")