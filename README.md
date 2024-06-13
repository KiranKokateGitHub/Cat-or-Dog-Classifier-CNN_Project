# Cat-or-Dog-Classifier-CNN_Project

This project implements a convolutional neural network (CNN) to classify images of dogs and cats. It utilizes the TensorFlow and Keras libraries for building and training the model. The dataset used for training and testing consists of images of dogs and cats.

## Dataset
The dataset used for this project can be found on Kaggle: Here is a link to dataset https://www.kaggle.com/datasets/salader/dogs-vs-cats . It contains a large number of images of dogs and cats for training and testing the classifier.

## Requirements
To run the code in this project, you need the following dependencies:

* Python (>= 3.6)

* TensorFlow (>= 2.0)

* NumPy

* Matplotlib

* OpenCV (cv2)


## Usage
1. Download the dataset from Kaggle using the provided link.
2. Ensure all dependencies are installed.
3. Run the provided code cells in a Python environment, such as Jupyter Notebook or Google Colab.

## Model Architecture
The CNN model architecture consists of several convolutional layers followed by max-pooling layers. Batch normalization and dropout layers are used for regularization. The model is compiled with the Adam optimizer and binary cross-entropy loss function.

## Training
The model is trained on the training dataset and validated on a separate validation dataset. Training progress and performance metrics are monitored during the training process.

## Evaluation
The trained model's performance is evaluated on the validation dataset using accuracy and loss metrics. Additionally, test images can be classified using the trained model.

## Results
The training process typically yields an accuracy of around 97-98% on the training dataset and around 75-80% on the validation dataset.

## Contributing
Contributions to this project are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
