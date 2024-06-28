# Anidex Image Classifier

Anidex Image Classifier is a Convolutional Neural Network model built using TensorFlow, Keras, NumPy, and Matplotlib libraries in Python. It classifies images of 90 different species from the animal kingdom.

## Overview

The Anidex Image Classifier project aims to classify images of various animal species using deep learning techniques. Inspired by the concept of the Pokédex from Pokémon, this model can predict the species of an animal from an input image.

## Model Details

- **Inspiration**: Inspired by the concept of Pokédex from Pokémon.
- **Validation Accuracy**: The model achieves a validation accuracy of 37.04%.
- **Animal Classes**: It can predict among 90 different animal species, including antelope, badger, bat, bear, and many others.
- **Architecture**
    - Convolutional layers (relu activation)
    - 2D-Maxpooling layers
    - Dropout layers (Dropout rate: 0.2)
    - Flattening layer
    - Dense layers for output
## Libraries Used

The model is implemented using the following Python libraries:
- [TensorFlow](https://tensorflow.org/): An open-source machine learning framework developed by Google that is used for building and training neural networks. Used to provide core functionalities for defining and training the Convolutional Neural Network model used in the project.
  
    ![Tensorflow](https://github.com/ISHOOO/Animal-Image-Classifier/assets/132544766/88315fbd-e7c7-42af-89d6-ce829dcdf616)
- [Keras](https://keras.io/): An open-source neural network library written in Python, designed to enable fast experimentation with deep neural networks. Used as a high-level API running on top of TensorFlow, simplifying the process of building and training the deep learning model.
  
    ![Keras](https://github.com/ISHOOO/Animal-Image-Classifier/assets/132544766/4f3fd801-eeef-4669-b8ff-2e0d59044279)
- [NumPy](https://numpy.org): A fundamental package for scientific computing in Python, providing support for arrays, matrices, and many mathematical functions. Used in the project for handling image data and performing various numerical operations required during data preprocessing and augmentation.
  
    ![Numpy](https://github.com/ISHOOO/Animal-Image-Classifier/assets/132544766/79cbee0c-a523-4474-b84a-f8fa41061de5)
- [Matplotlib](https://matplotlib.org/): Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It has been used in the project to visualize the training process, such as plotting training and validation accuracy, and displaying images during prediction.

    ![matplotlib](https://github.com/ISHOOO/Animal-Image-Classifier/assets/132544766/6d0553df-b48e-4889-b498-fb6fcff2b06b)

## Features

- **Data Augmentation**: Utilizes data augmentation techniques to increase the diversity of training data and improve model generalization.
- **Callbacks**: Implements callbacks such as Learning Rate Scheduler and Early Stopping to optimize model training and prevent overfitting.
- **Optimizer**: Adam optimizer is used for optimizing model parameters.
- **Loss Function**: Sparse Categorical Crossentropy is employed as the loss function for multi-class classification.


## Usage

To predict on an image using the Anidex Image Classifier:
1. Ensure Python dependencies are installed
```shell 
pip install numpy matplotlib tensorflow
```
2. Download `anidex.keras` and `predict.py` from the repository.
3. Update the variable `path_to_img` in `predict.py` with the relative path of the image you want to predict.
4. Run `predict.py` to perform predictions on unseen data.

To modify this project:
1. Clone this repository:
```shell
git clone "https://github.com/ISHOOO/Anidex-Image-Classifier.git"
```
2. Ensure Python dependencies are installed
```shell 
pip install numpy matplotlib tensorflow
```


## Visualizations

### Training and Validation Accuracy Comparison
  ![Training vs Validation Accuracy](https://github.com/ISHOOO/Animal-Image-Classifier/assets/132544766/affaa183-71e6-4d14-a221-880c472b66eb)

### Performing Prediction on Unseen Data
  ![Prediction Example 1](https://github.com/ISHOOO/Animal-Image-Classifier/assets/132544766/17728bee-7c3d-449a-9bdf-44e5f120c43d)
  ![Prediction Example 2](https://github.com/ISHOOO/Animal-Image-Classifier/assets/132544766/ceae5bc5-f85e-49d6-b72b-7f7c0f2151f3)


## Files Included
- `data_split.py`: Python script to split the 'animals data' directory into 'train' and 'valid' directories by 75% and 25% respectively.
- `anidex.keras`: Pre-trained model weights file.
- `predict.py`: Python script to perform predictions on new images.
- `unseen test data`: unseen images taken from an external source to test the generalizability of the model
- `Animal_recognition.ipynb`: Jupyter notebook containing script for model training

## Dataset

The dataset used for training and validation can be found on Kaggle:
- [Animal Image Dataset - 90 Different Animals](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

## List of animals which can be predicted by the model:
[antelope, 
badger, 
bat, 
bear, 
bee, 
beetle, 
bison, 
boar, 
butterfly, 
cat, 
caterpillar, 
chimpanzee, 
cockroach, 
cow, 
coyote, 
crab, 
crow, 
deer, 
dog, 
dolphin, 
donkey, 
dragonfly, 
duck, 
eagle, 
elephant, 
flamingo, 
fly, 
fox, 
goat, 
goldfish, 
goose, 
gorilla, 
grasshopper, 
hamster, 
hare, 
hedgehog, 
hippopotamus, 
hornbill, 
horse, 
hummingbird, 
hyena, 
jellyfish, 
kangaroo, 
koala, 
ladybugs, 
leopard, 
lion, 
lizard, 
lobster, 
mosquito, 
moth, 
mouse, 
octopus, 
okapi, 
orangutan, 
otter, 
owl, 
ox, 
oyster, 
panda, 
parrot, 
pelecaniformes, 
penguin, 
pig, 
pigeon, 
porcupine, 
possum, 
raccoon, 
rat, 
reindeer, 
rhinoceros, 
sandpiper, 
seahorse, 
seal, 
shark, 
sheep, 
snake, 
sparrow, 
squid, 
squirrel, 
starfish, 
swan, 
tiger, 
turkey, 
turtle, 
whale, 
wolf, 
wombat, 
woodpecker, 
zebra]


Feel free to explore and contribute to the Anidex Image Classifier project!
