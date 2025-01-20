# Data Quality Report

The model was trained using the MNIST dataset of handwritten digits. This dataset contains 60,000 images for training purposes and 10,000 additional images for testing and model evaluation. To improve the model performance the dataset was extended through the use of several data augmentation techniques.

The distribution of the digit classes 0-9 is even on average although the specific number of images for each class in the dataset varies slightly.

## Challenges faced while preparing the data

The MNIST dataset of handwritten digits provides a great basis for training and evaluating a Convolutional Neural Network (CNN) that performs image recognition. By applying data augmentation techniques, the dataset was extended to provide even more variety and therefore rigidity in the model's predictions. The biggest challenge in this process was defining and implementing the data augmentation. The following techniques were applied:
- Image shifting: shifting the images by a certain amount of pixels in one direction
- Adding noise: adding noise to the images
- Image blurring: blurring the images with a gaussian filter
Using these techniques, 6 new variations of each image in the training set could be created to provide more variety in the training data. Similarly, the testing data was augmented using the same techniques.

## Measures for data integrity

To maintain data integrity for the recognition of handwritten digits, it is crucial to ensure that the distribution of the 10 classes in the data is more or less equal. This is already given in the MNIST dataset. Since the same amount of image variations was created for each image in the dataset during the process of data augmentation, this property was also maintained in the extended dataset.

Additionally, the data exploration step in the pipeline can always be used to evaluate the distribution of the dataset and check if there is any change in data quality.
