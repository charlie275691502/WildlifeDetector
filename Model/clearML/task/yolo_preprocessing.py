import os, sys
import zipfile
from clearml import Dataset, Task
from collections import Counter
import os
import zipfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from main.preprocess import load_images_from_folders, make_subsets, downsample_images

def count_images(img_dir):
    img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    return len(img_files)

def count_labels(lbl_dir):
    lbl_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
    return len(lbl_files)

def count_bb(lbl_dir):
    total_bboxes = 0
    class_counter = Counter()
    lbl_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
    for lbl_file in lbl_files:
        with open(os.path.join(lbl_dir, lbl_file), "r") as f:
            lines = f.readlines()
            total_bboxes += len(lines)

            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(float(parts[0]))  # YOLO labels start with class_id
                    class_counter[class_id] += 1

    return total_bboxes, class_counter

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="WildlifeDetector", task_name="pipeline_yolo_preprocessing")

# program arguments
# Use either dataset_task_id to point to a tasks artifact or
# use a direct url with dataset_url
image_size = (640, 512)
args = {
    'dataset_task_id': '', #update id if it needs running locally
    'dataset_url': '',
    'random_state': 42,
    'target_count': 2000,
    'image_size': image_size,
}

# store arguments, later we will be able to change them from outside the code
task.connect(args)
print('Arguments: {}'.format(args))

# only create the task, we will actually execute it later
task.execute_remotely()

# Replace this with your dataset name/project
dataset_name = "Yolo"
dataset_project = "Wildlife"

# Get the dataset object
dataset = Dataset.get(dataset_name=dataset_name, dataset_project=dataset_project)
print(f"Found dataset: {dataset_name} in project {dataset_project}")

# Get local copy of the dataset (downloads if not already cached)
artifact_path = dataset.get_local_copy()
print("Dataset downloaded to:", artifact_path)

zip_path = os.path.join(artifact_path, "Yolo.zip")

# Unzip if needed

unzip_path = os.path.join(artifact_path, "unzipped")
os.makedirs(unzip_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_path)

print("Unzipped dataset saved to:", unzip_path)

# X, y, label_names = load_images_from_folders(os.path.join(unzip_path, "balanced_dataset"), image_size=args['image_size'])
# print(f"Loaded {len(X)} images with labels: {len(label_names)}")
# X, y = downsample_images(X, y, label_names, args['target_count'], args['random_state'])
# print(f"Balanced {len(X)} images with labels: {len(label_names)}")
# X_train, X_val, X_test, y_train, y_val, y_test = make_subsets(X, y, args['random_state'])

# === Upload Processed Data ===
# task.upload_artifact('X_train', X_train)
# task.upload_artifact('X_val', X_val)
# task.upload_artifact('X_test', X_test)
# task.upload_artifact('y_train', y_train)
# task.upload_artifact('y_val', y_val)
# task.upload_artifact('y_test', y_test)
# task.upload_artifact('label_names', label_names)
# task.upload_artifact('image_size', image_size)

# print("Artifacts uploaded in background.")

train_image_dir = os.path.join(unzip_path, "Yolo", "Train", "images")
train_label_dir = os.path.join(unzip_path, "Yolo", "Train", "labels")

valid_image_dir = os.path.join(unzip_path, "Yolo", "Validation", "images")
valid_label_dir = os.path.join(unzip_path, "Yolo", "Validation", "labels")

test_image_dir = os.path.join(unzip_path, "Yolo", "Test", "images")
test_label_dir = os.path.join(unzip_path, "Yolo", "Test", "labels")


# === Print stats ===
print("Train set:")
total_bboxes, class_counter = count_bb(train_label_dir)
print(f"Total images: {count_images(train_image_dir)}")
print(f"Total label files: {count_labels(train_label_dir)}")
print(f"Total bounding boxes: {total_bboxes}")
print("Class distribution:", class_counter)
print("\n")

print("Valid set:")
total_bboxes, class_counter = count_bb(valid_label_dir)
print(f"Total images: {count_images(valid_image_dir)}")
print(f"Total label files: {count_labels(valid_label_dir)}")
print(f"Total bounding boxes: {total_bboxes}")
print("Class distribution:", class_counter)
print("\n")

print("Test set:")
total_bboxes, class_counter = count_bb(test_label_dir)
print(f"Total images: {count_images(test_image_dir)}")
print(f"Total label files: {count_labels(test_label_dir)}")
print(f"Total bounding boxes: {total_bboxes}")
print("Class distribution:", class_counter)

print("Done.")
