import os
import random
import shutil
import numpy as np
import uuid
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

def is_valid_image(path):
    """
    Checks if an image is valid and readable using PIL.
    Returns True if valid, False if corrupted or unreadable.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # Only checks integrity
        return True
    except Exception:
        return False

def downsample_images(X, y, label_names, target_count, seed):
    """
    Downsamples or keeps only valid images from X and y.
    Ensures no class has more than target_count images.
    Returns the updated X and y.
    """
    # Group images by class
    class_images = {label: [] for label in range(len(label_names))}
    for img, label in zip(X, y):
        class_images[label].append(img)

    # Process each class
    updated_X = []
    updated_y = []

    for label, images in class_images.items():
        label_name = label_names[label]

        # Downsample or copy images
        if len(images) <= target_count:
            print(f"[{label_name}] Keeping all {len(images)} images.")
            updated_X.extend(images)
            updated_y.extend([label] * len(images))
        else:
            print(f"[{label_name}] Downsampling from {len(images)} to {target_count} images.")
            random.seed(seed)
            selected_images = random.sample(images, target_count)
            updated_X.extend(selected_images)
            updated_y.extend([label] * target_count)

    return np.array(updated_X), np.array(updated_y)

def load_images_from_folders(folder_path, image_size):
    """
    Loads images from subfolders and returns X (image array), y (class labels), and class names.
    """

    X = []
    y = []

    # List of class folders
    label_names = sorted(os.listdir(folder_path))  # Sorted for consistent label ordering

    for label, label_name in enumerate(label_names):
        label_folder = os.path.join(folder_path, label_name)

        if not os.path.isdir(label_folder):
            continue

        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)

            try:
                # Loads and resizes images into specified image size 
                # Converts images to RGB (i.e. pixel values are b/w 0 to 255), and stores images and labels into arrays 
                img = load_img(img_path, target_size=image_size, color_mode='rgb') 
                img_array = img_to_array(img)
                X.append(img_array)
                y.append(label)
            except Exception as e:
                print(f"Skipped {img_path} due to error: {e}")
                continue

    # Print statements for checking 
    print("\n")
    print(f"Length of X is {len(X)}")
    print(f"Length of y is {len(y)}")

    return np.array(X), np.array(y), label_names

def make_subsets(X, y, seed):

    """
    This function splits the data into training, validation and testing subsets with 80% of the data for training, 10% for validation and 10% for testing.
    """

    # First, split the data into training and temp (i.e. testing and validation)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.8, random_state=seed, stratify=y)

    # Then split the temp set into validation and testing
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=seed, stratify=y_temp)

    # Print statements for checking 
    print("\n")
    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:  ", y_val.shape)
    print("y_test shape: ", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_generators(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):

    """
    This function creates generators for the training, validation and testing data.
    """

    # Data augmentation on training data
    datagen = ImageDataGenerator(
        rotation_range= 40,
        width_shift_range= 0.2,
        height_shift_range= 0.2,
        shear_range= 0.2,
        zoom_range= 0.2,
        horizontal_flip= True,
        rescale=1./255,               # Rescaling pixel values to [0, 1]
        brightness_range= [0.5, 1.5]  # Augmentation (brightness)
    )

    train_generator = datagen.flow(
        X_train,                      # Numpy array of training images
        y_train,                      # Numpy array of integer labels  
        batch_size=batch_size,        # Batch size of 32
    )

    val_generator = datagen.flow(
        X_val, 
        y_val,                      
        batch_size=batch_size,     
    )

    test_generator = datagen.flow(
        X_test, 
        y_test,         
        batch_size=batch_size, 
        shuffle = False             
    )

    return train_generator, val_generator, test_generator



