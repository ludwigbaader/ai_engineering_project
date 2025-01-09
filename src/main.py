# Image recognition model

# Task: 
# Develop a model for image recognition

# Key considerations:
# - Data Augmentation: Enhance the dataset's diversity and robustness through augmentation techniques.
# - Model Architecture: Select or design a convolutional neural network (CNN) for image classification.
# - Evaluation Metrics: Use appropriate metrics like accuracy, precision, and recall for image-related tasks.
# - SE Best Practices: Follow SE best practices for code quality, including modularization and version control.

from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

from data_augmentation import augment_data


def main():
    # load the mnist dataset
    (img_train, lbl_train), (img_test, lbl_test) = mnist.load_data()
    print("MNIST dataset loaded successfully.")

    #explore_data(img_train, lbl_train, img_test, lbl_test)

    augment_data(img_train, lbl_train, img_test, lbl_test)


def explore_data(img_train: np.uint8, lbl_train: np.uint8, img_test: np.uint8, lbl_test: np.uint8):
    # Data Exploration: Understand the dataset's characteristics, such as size, class distribution, and data types.

    # Print 4 images in a row
    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(img_train[i+10], cmap='gray')
        plt.title(f"Label: {lbl_train[i+10]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def build_model():
    # Model Architecture: Select or design a convolutional neural network (CNN) for image classification.
    pass


def evaluate_model():
    # Evaluation Metrics: Use appropriate metrics like accuracy, precision, and recall for image-related tasks.
    pass


if __name__ == "__main__":
    main()
