import time
import keras
import pandas as pd
from main.evaluation_metrics import calculate_metrics, plot_confusion_matrix, plot_history
from main.model import build_model, train_model
from main.preprocess import create_generators, load_images_from_folders, make_subsets

if __name__ == "__main__":
    start_time = time.time()
    
    original_path = '../Data/garbage-dataset' # =========== EDIT THIS FOR GOOGLE ENTERPRISE

    # Defining image size and batch size
    image_size = (224, 224)
    batch_size = 32

    # Loading images from original directory
    X, y, label_names = load_images_from_folders(original_path, image_size)

    # Initialising integer labels i.e. 0, 1, 2 for waste categories
    int_labels = [i for i in range(len(label_names))]

    # Creating trainng, validation and testing subsets
    X_train, X_val, X_test, y_train, y_val, y_test = make_subsets(X, y)

    # Creating generators
    train_generator, val_generator, test_generator = create_generators(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)
    
    # Defining the best model
    best_model_file = "best_model.keras"

    # Training the model
    model = build_model(image_size, len(label_names))
    history = train_model(model, train_generator, val_generator, best_model_file)

    # Saving history 
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("training_history.csv", index=False)

    # Loading history and best model 
    history_df = pd.read_csv("training_history.csv")
    model = keras.models.load_model(best_model_file)
    
    # Evaluating metrics and plotting confusion matrix
    y_pred = calculate_metrics(test_generator, y_test, model, label_names)
    plot_confusion_matrix(y_test, y_pred, label_names, int_labels)

    # Plotting history 
    plot_history(history_df)
