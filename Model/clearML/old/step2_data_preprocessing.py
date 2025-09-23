import os, sys
import pickle
import zipfile
from clearml import Task

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.preprocess import load_images_from_folders, make_subsets, downsample_images


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="WildlifeDetector", task_name="Preprocess dataset")

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

# get dataset from task's artifact
if args['dataset_task_id']:
    dataset_upload_task = Task.get_task(task_id=args['dataset_task_id'])
    print(f"Found dataset in task: {args['dataset_task_id']}")
    print("Available artifacts:", dataset_upload_task.artifacts.keys())

    # Assuming the artifact name was 'my_dataset' and is a folder path
    artifact_path = dataset_upload_task.artifacts['dataset'].get_local_copy()
else:
    raise ValueError("Missing dataset_task_id!")

unzip_path = os.path.join(task.cache_dir, "unzipped_dataset")
os.makedirs(unzip_path, exist_ok=True)

with zipfile.ZipFile(artifact_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_path)

print("Dataset extracted to:", unzip_path)

X, y, label_names = load_images_from_folders(os.path.join(unzip_path, "balanced_dataset"), image_size=args['image_size'])
print(f"Loaded {len(X)} images with labels: {len(label_names)}")
X, y = downsample_images(X, y, label_names, args['target_count'], args['random_state'])
print(f"Balanced {len(X)} images with labels: {len(label_names)}")
X_train, X_val, X_test, y_train, y_val, y_test = make_subsets(X, y, args['random_state'])

# === Upload Processed Data ===
task.upload_artifact('X_train', X_train)
task.upload_artifact('X_val', X_val)
task.upload_artifact('X_test', X_test)
task.upload_artifact('y_train', y_train)
task.upload_artifact('y_val', y_val)
task.upload_artifact('y_test', y_test)
task.upload_artifact('label_names', label_names)
task.upload_artifact('image_size', image_size)

print("Artifacts uploaded in background.")
print("Done.")
