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
4. Run `predict.py` to perform predictions on unseen data.<br>
 __OR__<br>
You can visit the following website:<br>
[Anidex](https://goofyishu-anidex.hf.space/)

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
1. antelope
2. badger
3. bat
4. bear
5. bee
6. beetle
7. bison
8. boar
9. butterfly
10. cat
11. caterpillar
12. chimpanzee
13. cockroach
14. cow
15. coyote
16. crab
17. crow
18. deer
19. dog
20. dolphin
21. donkey
22. dragonfly
23. duck
24. eagle
25. elephant
26. flamingo
27. fly
28. fox
29. goat
30. goldfish
31. goose
32. gorilla
33. grasshopper
34. hamster
35. hare
36. hedgehog
37. hippopotamus
38. hornbill
39. horse
40. hummingbird
41. hyena
42. jellyfish
43. kangaroo
44. koala
45. ladybugs
46. leopard
47. lion
48. lizard
49. lobster
50. mosquito
51. moth
52. mouse
53. octopus
54. okapi
55. orangutan
56. otter
57. owl
58. ox
59. oyster
60. panda
61. parrot
62. pelecaniformes
63. penguin
64. pig
65. pigeon
66. porcupine
67. possum
68. raccoon
69. rat
70. reindeer
71. rhinoceros
72. sandpiper
73. seahorse
74. seal
75. shark
76. sheep
77. snake
78. sparrow 
79. squid
80. squirrel
81. starfish 
82. swan 
83. tiger 
84. turkey 
85. turtle 
86. whale 
87. wolf
88. wombat
89. woodpecker
90. zebra


Feel free to explore and contribute to the Anidex Image Classifier project!
