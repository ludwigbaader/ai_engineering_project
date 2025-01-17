# Image recognition model

# Task: 
# Develop a model for image recognition

# Key considerations:
# - Data Augmentation: Enhance the dataset's diversity and robustness through augmentation techniques.
# - Model Architecture: Select or design a convolutional neural network (CNN) for image classification.
# - Evaluation Metrics: Use appropriate metrics like accuracy, precision, and recall for image-related tasks.
# - SE Best Practices: Follow SE best practices for code quality, including modularization and version control.

import os
from keras import datasets, utils
import tensorflow as tf

from data_augmentation import augment_data
from data_exploration import explore_data
from model_architecture import build_model, train_model, evaluate_model, load_model


def main(model: str = ""):
    # load the mnist dataset
    (img_train, lbl_train), (img_test, lbl_test) = datasets.mnist.load_data()
    print("MNIST dataset loaded successfully.\n")

    print("Original Dataset of handwritten digits:")
    explore_data(img_train, lbl_train, img_test, lbl_test, show_plots=False)

    (img_aug_train, lbl_aug_train), (img_aug_test, lbl_aug_test) = augment_data(img_train, lbl_train, img_test, lbl_test)

    print("Augmented Dataset of handwritten digits:")
    explore_data(img_aug_train, lbl_aug_train, img_aug_test, lbl_aug_test, show_plots=False)

    # convert data to one-hot encoding
    lbl_aug_train_one_hot = utils.to_categorical(lbl_aug_train, num_classes=10)
    lbl_aug_test_one_hot = utils.to_categorical(lbl_aug_test, num_classes=10)

    # no model specified, create and train a new model
    if model == "":
        cnn_model = build_model("cnn_v02", location="models")
        training_history = train_model(
            cnn_model, 
            img_aug_train, lbl_aug_train_one_hot, 
            img_aug_test, lbl_aug_test_one_hot, 
            epochs=10, 
            checkpoint_directory=os.path.join("models", "cnn_v02")
        )
    
    # load the specified model
    else:
        cnn_model = load_model(model)
    
    test_loss, test_acc, test_precision, test_recall = evaluate_model(cnn_model, img_aug_test, lbl_aug_test_one_hot)
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}, Test precision: {test_precision}, Test recall: {test_recall}")


if __name__ == "__main__":
    #main() # trains and evaluates a new model
    main("cnn_v02") # loads and evaluate3s the pre-trained model cnn_v02
