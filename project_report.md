# AI Engineering Project Report - Creating a Convolutional Neural Network for handwritten digit recognition

The goal of this project is to create, train and deploy a Convolutional Neural Network (CNN) for the recognition of handwritten digits.

## Pipeline

### Data Pipeline

This model only uses the MNIST dataset of handwritten digits, there is no predefined pipeline for ingesting new data into the training and testing sets. If new data was to be injected, additional preprocessing steps such as ensuring all images are scaled to the correct dimensions (28x28 pixels) and have the correct format (single channel values between 0 and 255, white digit on black background). Furthermore, using the existing data exploration implementation an even distribution of data accross the 10 classes needs to be enforced.

### Model Pipeline

A more sophisticated pipeline exists to experiment with different model architectures and structures. When changing the model structure, as long as a new version number is used, an alternative model can be created, trained and saved. All of the saved models and even their checkpoints during training can be loaded and used again at any point. This enables experimenting with different CNNs or even entirely other Neural Network (NN) architectures.

### Evaluation Pipeline

The model evaluation provides tools to quickly evaluate the most important performance metrics of the CNN including test loss, test accuracy, test precision and test recall. These evaluations can, in combination with the ability to load various versions of the model, be used to quickly gain an insight into the models' performances and compare them across varying test datasets (for example including or excluding data augmentation).

### Deployment Pipeline

The deployment pipeline is designed to provide a quick and easy way of publishing the model to the huggingface platform. Since huggingface has tight regulations on how binary files such as the saved weights of the trained models are supposed to be handled when uploaded to the platform, the repository can not be published straight away. Instead, the relevant files such as the configuration and weights of a new model, need to be added to the mirrored repository in the `huggingface` subfolder and uploaded to huggingface using the git large file system (LFS). Once uploaded, the model can be publicly viewed and used on huggingface.
