import numpy as np
from scipy.ndimage import shift


def augment_data(img_train: np.uint8, lbl_train: np.uint8, img_test: np.uint8, lbl_test: np.uint8) -> tuple[tuple[np.uint8, np.uint8], tuple[np.uint8, np.uint8]]:
    # Data Augmentation: Enhance the dataset's diversity and robustness through augmentation techniques.

    # augmentation techniques:
    # - shifting the image by a few pixels
    # - adding noise to the image
    # - synthetic data generation -> using a GAN to generate new images

    img_train_shifted, lbl_train_shifted = augment_images_shift(img_train, lbl_train, 1)
    img_test_shifted, lbl_test_shifted = augment_images_shift(img_test, lbl_test, 1)

    return ((img_train, lbl_train), (img_test, lbl_test))


def augment_images_shift(images, labels, amount: int) -> tuple[np.uint8, np.uint8]:
    shifted_images = []
    shifted_labels = []

    directions = ['left', 'right', 'up', 'down']

    for img in images:
        for direction in directions:
            shifted_image = shift_image(img, direction, amount)
            shifted_images.append(shifted_image)
            shifted_labels.append(labels)

    return (shifted_images, shifted_labels)

def shift_image(image: np.uint8, direction: str, amount: int) -> np.uint8:
    # shift the given image by the specified amount of pixels in the specified direction
    if direction == 'left':
        return shift(image, [0, -amount], cval=0)
    elif direction == 'right':
        return shift(image, [0, amount], cval=0)
    elif direction == 'up':
        return shift(image, [-amount, 0], cval=0)
    elif direction == 'down':
        return shift(image, [amount, 0], cval=0)


def augment_images_noise(images, labels, amount: float) -> tuple[np.uint8, np.uint8]:
    # Add noise to the images
    # add a random number to each pixel value
    for img in images:
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                img[x, y] += np.random.uniform(-128, 128) * amount
