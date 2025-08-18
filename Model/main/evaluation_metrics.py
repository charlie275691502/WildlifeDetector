
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_history(history, save_path="accuracy.jpg"):
    """
    This function plots the training and validation accuracy and loss.
    """

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], "bo", label="Training accuracy")
    plt.plot(history['val_accuracy'], "b", label="Validation accuracy")
    plt.title("Training and validation accuracy", fontsize = 16)
    plt.xlabel("Epochs", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.xlim([0, len(history)])  
    plt.xticks(range(len(history)))  
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], "bo", label="Training loss")
    plt.plot(history['val_loss'], "b", label="Validation loss")
    plt.title("Training and validation loss", fontsize = 16)
    plt.xlabel("Epochs", fontsize = 12)
    plt.ylabel("Loss", fontsize = 12)
    plt.xlim([0, len(history)])  
    plt.xticks(range(len(history)))  
    plt.legend()


    plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')

    plt.show()

def calculate_metrics(test_generator, y_test, model, label_names):
    """
    This function calculates the evaluation metrics - accuracy, precision, recall, F1 score.
    """    

    # Predicting class probabilities
    y_pred_probs = model.predict(test_generator)

    # Converting probabilities to class predictions
    y_pred = np.argmax(y_pred_probs, axis=1)  

    # Computing evaluation metrics per class
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)

    # Converting to a DataFrame for better readability
    df_report = pd.DataFrame(report).transpose()

    # Display the table
    print("\n==================== Evaluation Metrics Report ====================")
    print(df_report)
    print("\n")

    # Saving to CSV file
    df_report.to_csv("evaluation_metrics.csv")

    return y_pred, df_report


def plot_confusion_matrix(y_true, y_pred, label_names, int_labels, save_path="confusion_matrix.jpg"):
    """
    This function plots the confusion matrix.
    """

    # Computing confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels = int_labels)

    # Plotting the confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", square=True, xticklabels= label_names, yticklabels= label_names)
    plt.xlabel("Predicted Labels", fontsize = 12)
    plt.ylabel("True Labels", fontsize = 12)
    plt.title("Confusion Matrix of DenseNet201 Classifier", fontsize = 14)

    plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

