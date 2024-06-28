from tensorflow.keras import utils, models
import tensorflow as tf
import numpy as np
from matplotlib import image as mpimg, pyplot as plt
#update this 👇variable to perform prediction on your image 
path_to_img='unseen test data\\antelope.png'
model= models.load_model('anidex.keras')
data_cat=['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']
image = utils.load_img(path_to_img, target_size=(240,240))
img_arr = utils.array_to_img(image)
img_bat=tf.expand_dims(img_arr,0)
predict = model.predict(img_bat, verbose=0)
score = tf.nn.softmax(predict)
plt.imshow(mpimg.imread('unseen test data\\antelope.png'))
plt.axis('off')
plt.text(0.5, -0.1, f"Prediction: {data_cat[np.argmax(score)]}", size=12, ha='center', transform=plt.gca().transAxes) # caption to the image
plt.show()