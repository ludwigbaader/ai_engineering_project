import matplotlib.pyplot as plt
import numpy as np


def explore_data(img_train: np.uint8, lbl_train: np.uint8, img_test: np.uint8, lbl_test: np.uint8, show_plots=True):
    '''Understand the dataset's characteristics, such as size, class distribution, and data types.'''

    # total number of entries in the training and testing datasets
    train_size = img_train.shape[0]
    test_size = img_test.shape[0]

    print(f"Training dataset size: {train_size}")
    print(f"Testing dataset size: {test_size}")

    # count the number of times each digit appears in the training and testing datasets
    train_counts = np.bincount(lbl_train)
    test_counts = np.bincount(lbl_test)

    train_counts_avg = np.mean(train_counts)
    train_counts_std = np.std(train_counts)

    test_counts_avg = np.mean(test_counts)
    test_counts_std = np.std(test_counts)

    print(f"Training dataset class distribution: {train_counts}, Mean: {train_counts_avg}, Std: {train_counts_std}")
    print(f"Testing dataset class distribution: {test_counts}, Mean: {test_counts_avg}, Std: {test_counts_std}")

    if show_plots:
        # Print the first 4 images in the training dataset
        plt.figure(figsize=(10, 5))
        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.imshow(img_train[i], cmap='gray')
            plt.title(f"Label: {lbl_train[i]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    print()
