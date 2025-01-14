# Image recognition model

# Task: 
# Develop a model for image recognition

# Key considerations:
# - Data Augmentation: Enhance the dataset's diversity and robustness through augmentation techniques.
# - Model Architecture: Select or design a convolutional neural network (CNN) for image classification.
# - Evaluation Metrics: Use appropriate metrics like accuracy, precision, and recall for image-related tasks.
# - SE Best Practices: Follow SE best practices for code quality, including modularization and version control.

from tensorflow.keras.datasets import mnist

from data_augmentation import augment_data
from data_exploration import explore_data


def main():
    # load the mnist dataset
    (img_train, lbl_train), (img_test, lbl_test) = mnist.load_data()
    print("MNIST dataset loaded successfully.")

    print("Original Dataset of handwritten digits:")
    explore_data(img_train, lbl_train, img_test, lbl_test, show_plots=False)

    (img_aug_train, lbl_aug_train), (img_aug_test, lbl_aug_test) = augment_data(img_train, lbl_train, img_test, lbl_test)

    print("Augmented Dataset of handwritten digits:")
    explore_data(img_aug_train, lbl_aug_train, img_aug_test, lbl_aug_test, show_plots=True)

    build_model()


def build_model():
    # Model Architecture: Select or design a convolutional neural network (CNN) for image classification.
    pass


def evaluate_model():
    # Evaluation Metrics: Use appropriate metrics like accuracy, precision, and recall for image-related tasks.
    pass


if __name__ == "__main__":
    main()
