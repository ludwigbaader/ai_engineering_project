import multiprocessing, time
import numpy as np
from scipy.ndimage import shift, gaussian_filter


def augment_data(img_train: np.uint8, lbl_train: np.uint8, img_test: np.uint8, lbl_test: np.uint8) -> tuple[tuple[np.uint8, np.uint8], tuple[np.uint8, np.uint8]]:
    '''Augment the training and testing datasets to enhance the dataset's diversity and robustness.'''

    # augmentation techniques:
    # - shifting the images by a few pixels
    # - adding noise to the images
    # - blurring the images

    print("Augmenting the training and testing datasets...")
    print("Generating new images through shifting...")

    img_train_shifted, lbl_train_shifted = augment_images_shift(img_train, lbl_train, 1)
    img_test_shifted, lbl_test_shifted = augment_images_shift(img_test, lbl_test, 1)

    print("Generating new images through noise addition...")

    start = time.time()
    img_train_noisy, lbl_train_noisy = augment_images_noise_threaded(img_train, lbl_train, 0.2)
    img_test_noisy, lbl_test_noisy = augment_images_noise_threaded(img_test, lbl_test, 0.2)
    duration = time.time() - start
    print(f"Time taken for adding noise to images: {duration:.3f} seconds")

    print("Generating new images through blurring...\n")

    img_train_blurred, lbl_train_blurred = augment_images_blur(img_train, lbl_train, 2)
    img_test_blurred, lbl_test_blurred = augment_images_blur(img_test, lbl_test, 2)

    img_train_augmented = np.concatenate((img_train, img_train_shifted, img_train_noisy, img_train_blurred))
    lbl_train_augmented = np.concatenate((lbl_train, lbl_train_shifted, lbl_train_noisy, lbl_train_blurred))
    img_test_augmented = np.concatenate((img_test, img_test_shifted, img_test_noisy, img_test_blurred))
    lbl_test_augmented = np.concatenate((lbl_test, lbl_test_shifted, lbl_test_noisy, lbl_test_blurred))

    return ((img_train_augmented, lbl_train_augmented), (img_test_augmented, lbl_test_augmented))


def augment_images_shift(images, labels, amount: int) -> tuple[np.uint8, np.uint8]:
    '''Shift the given images by the specified amount of pixels in four directions (left, right, up, down). Returns four times the amount of provided images.'''

    shifted_images = []
    new_labels = []

    directions = ['left', 'right', 'up', 'down']

    for direction in directions:
        for img in images:
            shifted_image = shift_image(img, direction, amount)
            shifted_images.append(shifted_image)
        new_labels.extend(labels)

    return (np.array(shifted_images), new_labels)

def shift_image(image: np.uint8, direction: str, amount: int) -> np.uint8:
    '''Shift the given image by the specified amount of pixels in the specified direction'''

    if direction == 'left':
        return shift(image, [0, -amount], cval=0)
    elif direction == 'right':
        return shift(image, [0, amount], cval=0)
    elif direction == 'up':
        return shift(image, [-amount, 0], cval=0)
    elif direction == 'down':
        return shift(image, [amount, 0], cval=0)


def augment_images_noise(images: np.uint8, labels: np.uint8, amount: float) -> tuple[np.uint8, np.uint8]:
    '''Add the speicified amount of noise to the given images. Returns the same amount of images than the provided ones.'''
    
    for img in images:
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                # add a random number to each pixel, ensuring the pixel value stays within the valid range [0, 255]
                img[x, y] = min(max(img[x, y] + np.random.uniform(-128, 128) * amount, 0), 255)
    
    return (images, labels)


def augment_images_noise_threaded(images: np.uint8, labels: np.uint8, amount: float, num_of_threads: int = 24) -> tuple[np.uint8, np.uint8]:
    '''Add the speicified amount of noise to the given images using multiple threads. Returns the same amount of images than the provided ones.'''

    # generate 24 threads
    threads = []
    for i in range(num_of_threads):
        img_batch = images[i * (len(images) // num_of_threads): (i + 1) * (len(images) // num_of_threads)]
        t = multiprocessing.Process(target=add_noise_to_images, args=(img_batch, amount))
        #t = threading.Thread(target=add_noise_to_images, args=(img_batch, amount))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return (images, labels)

def add_noise_to_images(images: np.uint8, amount: float):
    for img in images:
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                # add a random number to each pixel, ensuring the pixel value stays within the valid range [0, 255]
                img[x, y] = min(max(img[x, y] + np.random.uniform(-128, 128) * amount, 0), 255)


def augment_images_blur(images: np.uint8, labels: np.uint8, amount: float) -> tuple[np.uint8, np.uint8]:
    '''Blur the given images using a Gaussian filter with the specified amount of sigma. Returns the same amount of images than the provided ones.'''

    blurred_images = []

    # Apply Gaussian blur to the images
    for img in images:
        blurred_image = gaussian_filter(img, sigma=amount)
        blurred_images.append(blurred_image)
    
    return (np.array(blurred_images), labels)
