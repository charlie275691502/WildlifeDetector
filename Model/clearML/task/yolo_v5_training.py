import os
from clearml import Task, Dataset
import subprocess
import yaml
import tempfile

# === Initialize ClearML task ===
Task.force_requirements_env_freeze(False)
task = Task.init(
    project_name="WildlifeDetector",
    task_name="pipeline_yolo_v5_training",
    output_uri=True  # Upload trained weights/artifacts automatically
)
task._update_requirements = False
task.set_packages([
    "torch==2.5.1",
    "torchvision==0.20.1",
    "PyYAML==6.0.2",
    "matplotlib==3.10.1",
    "numpy==2.2.4",
    "pandas==2.2.3",
    "pillow==11.1.0",
    "seaborn==0.13.2",
    "clearml==2.0.2",
])

# === YOLOv5 repo path (adjust to your local repo folder) ===
YOLOV5_REPO = os.path.abspath("Model/yolov5")  # adjust if your repo is in a subfolder
TRAIN_SCRIPT = os.path.join(YOLOV5_REPO, "train.py")
VAL_SCRIPT = os.path.join(YOLOV5_REPO, "val.py")


# === Fetch dataset via ClearML Dataset ===
dataset = Dataset.get(dataset_name="Yolo", dataset_project="Wildlife")
dataset_path = dataset.get_local_copy()  # folder with Yolo.zip and Yolo.yaml

# Path to the unzipped dataset
unzip_path = os.path.join(dataset_path, "unzipped")

# Build dataset.yaml dynamically
data_dict = {
    "train": os.path.join(unzip_path, "Yolo", "Train", "images"),
    "val": os.path.join(unzip_path, "Yolo", "Validation", "images"),
    "test": os.path.join(unzip_path, "Yolo", "Test", "images"),
    "nc": 3,  # number of classes
    "names": ["deer", "fox", "boar"]  # replace with your class list
}

# Save to a temporary YAML
tmp_yaml = os.path.join(tempfile.gettempdir(), "dataset_clearml.yaml")
with open(tmp_yaml, "w") as f:
    yaml.safe_dump(data_dict, f)

# === Define parameters (can be changed from UI) ===
args = {
    "img_size": 640,
    "batch_size": 8,
    "epochs": 3,
    "cfg_yaml": os.path.join(YOLOV5_REPO, "models", "yolov5m.yaml"), # Relative path to model config
    "weights": "yolov5m.pt",
    "name": "yolov5_custom",
    "cache": True,
    "patience": 30,
}

task.connect(args)  # Makes these params editable in ClearML UI

# only create the task, we will actually execute it later
task.execute_remotely()

# === Paths ===
# Assuming yolov5 repo is in the same directory as this script


# === Step 1: Training ===
train_cmd = [
    "python", TRAIN_SCRIPT,
    "--img", str(args["img_size"]),
    "--batch", str(args["batch_size"]),
    "--epochs", str(args["epochs"]),
    "--data", tmp_yaml,
    "--cfg", args["cfg_yaml"],
    "--weights", args["weights"],
    "--name", args["name"],
    "--patience", str(args["patience"]),
    "--workers", "0",            # Windows-friendly
    "--project", "runs_clearml"  # shorter path
]

if args["cache"]:
    train_cmd.append("--cache")

print("Running YOLOv5 training...")
subprocess.run(train_cmd, check=True)
print("Training finished!")

# === Step 2: Validation ===
weights_path = os.path.join("runs_clearml", args["name"], "weights", "best.pt")
val_cmd = [
    "python", VAL_SCRIPT,
    "--weights", weights_path,
    "--data", tmp_yaml,
    "--task", "test"
]

print("Running YOLOv5 validation...")
subprocess.run(val_cmd, check=True)
print("Validation finished!")

# === Upload trained model as artifact ===
task.upload_artifact(name="trained_model", artifact_object=weights_path)
print(f"Model uploaded to ClearML: {weights_path}")