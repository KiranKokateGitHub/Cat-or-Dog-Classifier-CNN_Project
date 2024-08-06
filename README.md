# Cat-or-Dog-Classifier-CNN_Project

This project implements a convolutional neural network (CNN) to classify images of dogs and cats. It utilizes the TensorFlow and Keras libraries for building and training the model. The dataset used for training and testing consists of images of dogs and cats.

## Dataset
The dataset used for this project can be found on Kaggle: Here is a link to dataset https://www.kaggle.com/datasets/salader/dogs-vs-cats . It contains a large number of images of dogs and cats for training and testing the classifier.
The dataset contains thousands of labeled images of cats and dogs, split into training and testing sets. Before using the dataset, please ensure you have accepted the competition rules on the Kaggle website.

## Model Architecture
The classification model utilizes a pre-trained MobileNet V2 architecture as the base feature extractor. MobileNet V2 is a lightweight and efficient network known for its performance on mobile and embedded vision applications. For this project:

* The MobileNet V2 model, loaded from TensorFlow Hub, serves as a fixed feature extractor.
* A custom dense layer is added on top to classify the extracted features into two categories: dog and cat.
The use of transfer learning with a pre-trained model helps in achieving better accuracy with less training time and computational resources.

## Training
The model is compiled using the Adam optimizer and Sparse Categorical Crossentropy as the loss function. The training process involves fine-tuning the dense layer, while the pre-trained layers of MobileNet V2 are kept frozen to leverage the learned feature representations.

## Evaluation
After training, the model is evaluated on a separate test dataset to measure its performance. The evaluation metrics include accuracy and loss. The model achieved an impressive test accuracy of approximately 97.75%, indicating that it can effectively distinguish between images of cats and dogs.

## Prediction System
The predictive system allows users to input an image and receive a classification result indicating whether the image contains a cat or a dog. This functionality can be used for various applications, such as automatic tagging of images or developing pet recognition systems.

## Requirements
To run the code in this project, you need the following dependencies:

* Python (>= 3.6)

* TensorFlow (>= 2.0)

* Keras

* NumPy

* Matplotlib

* OpenCV (cv2)


## Usage
1. **Data Preparation**: Download and prepare the dataset from Kaggle.
2. **Model Training**: Train the model using the provided scripts, ensuring that the dataset is properly split into training and testing sets.
3. **Evaluation**: Evaluate the model's performance and fine-tune as necessary.
4. **Prediction**: Use the predictive system to classify new images of cats and dogs.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to enhance the project.
