# Image recognition model

# Task: 
# Develop a model for image recognition

# Key considerations:
# - Data Augmentation: Enhance the dataset's diversity and robustness through augmentation techniques.
# - Model Architecture: Select or design a convolutional neural network (CNN) for image classification.
# - Evaluation Metrics: Use appropriate metrics like accuracy, precision, and recall for image-related tasks.
# - SE Best Practices: Follow SE best practices for code quality, including modularization and version control.

import os, json
import tensorflow as tf
from keras import datasets, layers, models, callbacks
import numpy as np

from data_augmentation import augment_data
from data_exploration import explore_data


def main(model: str = ""):
    # load the mnist dataset
    (img_train, lbl_train), (img_test, lbl_test) = datasets.mnist.load_data()
    print("MNIST dataset loaded successfully.\n")

    print("Original Dataset of handwritten digits:")
    explore_data(img_train, lbl_train, img_test, lbl_test, show_plots=False)

    (img_aug_train, lbl_aug_train), (img_aug_test, lbl_aug_test) = augment_data(img_train, lbl_train, img_test, lbl_test)

    print("Augmented Dataset of handwritten digits:")
    explore_data(img_aug_train, lbl_aug_train, img_aug_test, lbl_aug_test, show_plots=False)

    # no model specified, create and train a new model
    if model == "":
        cnn_model = build_model("cnn_v01", location="models")
        training_history = train_model(
            cnn_model, 
            img_aug_train, lbl_aug_train, 
            img_aug_test, lbl_aug_test, 
            epochs=10, 
            checkpoint_directory=os.path.join("models", "cnn_v01")
        )
    
    # load the specified model
    else:
        cnn_model = load_model(model)
    
    test_loss, test_acc = evaluate_model(cnn_model, img_aug_test, lbl_aug_test)
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")


def build_model(name: str, location: str = "models") -> models.Sequential:
    # Model Architecture: Select or design a convolutional neural network (CNN) for image classification.
    print(f"Building model: {name}")

    # Model structure based on: https://www.tensorflow.org/tutorials/images/cnn
    model = models.Sequential()
    # first layer - convolutional layer with 32 filters, 3x3 kernel
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # second layer - max pooling layer with 2x2 pool size
    model.add(layers.MaxPooling2D((2, 2)))
    # third layer - convolutional layer with 64 filters, 3x3 kernel
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # fourth layer - max pooling layer with 2x2 pool size
    model.add(layers.MaxPooling2D((2, 2)))
    # fifth layer - convolutional layer with 64 filters, 3x3 kernel
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # sixth layer - flatten convolutional output and feed into dense layer with 64 units
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # seventh layer - output layer with 10 units (one for each digit)
    model.add(layers.Dense(10))

    #model.summary()

    # compile the model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    print("Model compiled successfully.")

    # save the model structure for logging of different model versions
    model_configuration = {
        "name": name,
        "model": json.loads(model.to_json()),
    }
    if not os.path.isdir(os.path.join(location, name)):
        os.makedirs(os.path.join(location, name))
    
    with open(os.path.join(location, name, "model_config.json"), "w") as f:
        f.write(json.dumps(model_configuration))
    
    print(f"Saved model configuration to {os.path.join(location, name, 'model_config.json')}\n")

    return model


def train_model(
        model: models.Sequential, 
        img_train: np.uint8, 
        lbl_train: np.uint8, 
        img_test: np.uint8, 
        lbl_test: np.uint8, 
        epochs=10, 
        checkpoint_directory: str = None
    ) -> callbacks.History:
    '''Train the model using the provided training dataset and evaluate it on the testing dataset. 
    Train for specified number of epochs and save checkpoints if a directory is provided.'''

    print(f"Training model for {epochs} epochs")

    if checkpoint_directory:
        # create a callback to save model checkpoints
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=os.path.join(
                checkpoint_directory, 
                "checkpoints", 
                f"{os.path.basename(checkpoint_directory)}_epoch_{{epoch:03d}}.weights.h5"
            ), 
            save_weights_only=True, 
            verbose=1
        )
        
        # train the model
        training_history = model.fit(
            img_train, lbl_train, 
            epochs=epochs, 
            validation_data=(img_test, lbl_test), 
            callbacks=[checkpoint_callback]
        )
    else:
        # train the model without saving checkpoints
        training_history = model.fit(
            img_train, lbl_train, 
            epochs=epochs, 
            validation_data=(img_test, lbl_test)
        )
    
    print(f"Model trained successfully. Checkpoints containing the weights saved to {os.path.join(checkpoint_directory, 'checkpoints')}\n")

    # Train the model using the training dataset
    return training_history


def evaluate_model(model: models.Sequential, img_test: np.uint8, lbl_test: np.uint8):
    # Evaluation Metrics: Use appropriate metrics like accuracy, precision, and recall for image-related tasks.
    print("Evaluating model on the testing dataset")

    test_loss, test_acc = model.evaluate(img_test,  lbl_test, verbose=2)

    print("Model evaluation complete.\n")

    return (test_loss, test_acc)


def load_model(name: str, location: str = "models") -> models.Sequential:
    # check if model exists
    if not name in os.listdir(location):
        print(f"Model {name} not found in {location}")
        return None

    # Load a model from the specified location
    f = open(os.path.join(location, name, "model_config.json"))
    model_config = json.loads(f.read())

    model: models.Sequential = models.model_from_json(json.dumps(model_config["model"]))
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Load the newest model weights
    version = 1
    for weight_file in os.listdir(os.path.join(location, name, "checkpoints")):
        if weight_file.endswith(".weights.h5"):
            weight_file_version = int(weight_file.split("_")[-1].replace(".weights.h5", ""))
            if weight_file_version > version:
                version = weight_file_version

    model.load_weights(os.path.join(location, name, "checkpoints", f"{name}_epoch_{str(version).zfill(3)}.weights.h5"))
    
    return model


if __name__ == "__main__":
    main("cnn_v01")
