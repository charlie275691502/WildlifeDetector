from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet201

def define_base_model():
    """
    This function defines the base model i.e DenseNet201 model
    """

    conv_base = DenseNet201(
        weights="imagenet",
        include_top=False,
    )

    # This empties the list of trainable weights i.e. freezing weights of convolutional base  
    # Convolutional layers in pre-trained models have learnt features e.g. edges, shapes etc. 
    # Therefore we extract these features from the convolutional base, to be used in tailored model 
    conv_base.trainable = False

    return conv_base

def build_model(image_size, num_classes, neural_count, learning_rate):
    """
    This function builds a convolutional neural network model.

    """

    # Defining the inputs as shape (224, 224, 3); 3 denotes the 3 colour channels i.e. RGB
    inputs = keras.Input(shape=(image_size[0], image_size[1], 3))  

    # Passing through the convolutional base
    conv_base = define_base_model()
    x = conv_base(inputs)

    # Reducing the entire feature map into single vector; reduces overfitting and has fewer parameter than Flatten
    x = layers.GlobalAveragePooling2D()(x)
    
    # Defining Dense and Batch Normalisation layers
    x = layers.Dense(neural_count)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)

    # Defining the outputs and the number of output classes
    outputs = layers.Dense(num_classes, activation = "softmax")(x) 

    # Defining the model 
    model = keras.Model(inputs, outputs)

    # Compiling the model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, 
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model

def train_model(model, train_generator, val_generator, best_model_file):
    """
    This function trains the model.
    """

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_file,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_generator,
        epochs=20, 
        validation_data=val_generator,
        callbacks=callbacks,
    )

    print(f"Model training complete. Best model saved to: {best_model_file}")
        
    return history








