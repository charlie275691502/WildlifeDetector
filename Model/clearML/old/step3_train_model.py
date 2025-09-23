import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from clearml import Task, Logger
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main.evaluation_metrics import calculate_metrics, plot_confusion_matrix, plot_history
from main.model import build_model, train_model
from main.preprocess import create_generators


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="WildlifeDetector", task_name="Train model")
logger = Logger.current_logger()

# Arguments
args = {
    'dataset_task_id': '', # replace the value only when you need debug locally
    'batch_size': 32,
    'learning_rate': 0.001,
    'neural_count': 256,
}
task.connect(args)

# only create the task, we will actually execute it later
task.execute_remotely() # After passing local testing, you should uncomment this command to initial task to ClearML

print('Retrieving dataset')
dataset_task = Task.get_task(task_id=args['dataset_task_id'])
X_train = dataset_task.artifacts['X_train'].get()
X_test = dataset_task.artifacts['X_test'].get()
X_val = dataset_task.artifacts['X_val'].get()
y_train = dataset_task.artifacts['y_train'].get()
y_test = dataset_task.artifacts['y_test'].get()
y_val = dataset_task.artifacts['y_val'].get()
label_names = dataset_task.artifacts['label_names'].get()
image_size = dataset_task.artifacts['image_size'].get()
print('Dataset loaded')


batch_size = args['batch_size']
learning_rate = args['learning_rate']
neural_count = args['neural_count']
# Creating generators
train_generator, val_generator, test_generator = create_generators(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)
int_labels = [i for i in range(len(label_names))]
# Defining the best model
best_model_file = "best_model.keras"

print('Generators loaded')

# Training the model
model = build_model(image_size, len(label_names), neural_count, learning_rate)
history = train_model(model, train_generator, val_generator, best_model_file)

for epoch in range(len(history.history['loss'])):
    task.get_logger().report_scalar(title="train", series="epoch_loss", value=history.history['loss'][epoch], iteration=epoch)
    task.get_logger().report_scalar(title="validation", series="accuracy", value=history.history['val_accuracy'][epoch], iteration=epoch)
    task.get_logger().report_scalar(title="validation", series="accuracy", value=history.history['val_accuracy'][epoch], iteration=0)

training_accuracy = history.history['accuracy'][-1]
validation_accuracy = history.history['val_accuracy'][-1]

print(f"Training Accuracy: {training_accuracy}")
print(f"Validation Accuracy: {validation_accuracy}")

model.save("model.keras")
task.upload_artifact(name="model", artifact_object="model.keras")

y_pred, df_report = calculate_metrics(test_generator, y_test, model, label_names)
save_path="confusion_matrix.jpg"
upload_name="confusion_matrix"
plot_confusion_matrix(y_test, y_pred, label_names, int_labels, save_path)

df_report.to_csv("evaluation_metrics.csv")
task.upload_artifact(name="evaluation_metrics", artifact_object="evaluation_metrics.csv")
task.upload_artifact(name=upload_name, artifact_object=save_path)

save_path="accuracy.jpg"
upload_name="accuracy"
history_df = pd.DataFrame(history.history)
plot_history(history_df, save_path)
task.upload_artifact(name=upload_name, artifact_object=save_path)

