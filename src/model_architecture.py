import os, json
import tensorflow as tf
from keras import models, layers, callbacks, metrics, losses
import numpy as np


def build_model(name: str, location: str = "models") -> models.Sequential:
    '''Create a new convolutional neural network for image classification. Stores the model in the specified location using the specified name.'''
    
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
    model.add(layers.Dense(10, activation='softmax'))

    # compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    print("Model compiled successfully.")

    model.summary()
    print()

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
    '''Evaluate the model on the testing dataset using appropriate metrics like accuracy and precision.'''

    # Evaluation Metrics: Use appropriate metrics like accuracy, precision, and recall for image-related tasks.
    print("Evaluating model on the testing dataset")

    test_loss, test_acc, test_precision, test_recall = model.evaluate(img_test,  lbl_test, verbose=2)

    print("Model evaluation complete.\n")

    return (test_loss, test_acc, test_precision, test_recall)


def load_model(name: str, location: str = "models") -> models.Sequential:
    '''Load a previously trained model from the specified location.'''

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
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    # Load the newest model weights
    version = 1
    for weight_file in os.listdir(os.path.join(location, name, "checkpoints")):
        if weight_file.endswith(".weights.h5"):
            weight_file_version = int(weight_file.split("_")[-1].replace(".weights.h5", ""))
            if weight_file_version > version:
                version = weight_file_version

    model.load_weights(os.path.join(location, name, "checkpoints", f"{name}_epoch_{str(version).zfill(3)}.weights.h5"))

    print(f"Loaded model {name} from {os.path.join(location, name)}:")
    model.summary()
    print()
    
    return model
