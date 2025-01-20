---
license: gpl-3.0
language:
- en
datasets:
- mnist
---

# Machine Learning Model for Image Recognition

This project has been created as part of the course "AI Engineering" at the University of Karlstad, Sweden.

This model uses multiple layers of Convolutional Neural Networks (CNNs) to perform image recognition of handwritten digits. The model is trained on the MNIST dataset of 60,000 images of handwritten digits. It is capable of predicting the number of the provided handwritten digit in the testset of 10,000 images with an accuracy of more than 98%.

## The training data

The training data is a modified version of the MNIST dataset of images of handwritten digits containing 60,000 entires. Through several data augmentation techniques this dataset has been extended to 420,000 images the model has been trained on.

## The model

The model consists of a CNN with multiple layers. Layers:
- 1st layer: Convolutional layer with 32 filters of the size 3x3
- 2nd layer: 2D max-pooling with a pool of the size 2x2
- 3rd layer: Convolutional layer with 64 filters of the size 3x3
- 4th layer: 2D max-pooling with a pool of the size 2x2
- 5th layer: Convolutional layer with 64 filters of the size 3x3
- 6th layer: Dense layer with 64 units
- 7th layer (output layer): Dense layer with 10 units

This architecture results in a total of 93,322 trainable parameters.
