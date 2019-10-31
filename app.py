import tensorflow as tf
from model import ClothingID
import numpy as np
from tensorflow import keras

def main():
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0
    predictions = ClothingID(
        training_images, training_labels, test_images, test_labels)
    response = np.argmax(predictions[42])
    if response == 0:
        print("The image is of a T-shirt/top")
    elif response == 1:
        print("The image is of a Trouser")
    elif response == 2:
        print("The image is of a Pullover")
    elif response == 3:
        print("The image is of a Dress")
    elif response == 4:
        print("The image is of a Coat")
    elif response == 5:
        print("The image is of a Sandal")
    elif response == 6:
        print("The image is of a Shirt")
    elif response == 7:
        print("The image is of a Sneaker")
    elif response == 8:
        print("The image is of a Bag")
    else:
        print("The image is of a Ankle Boot")

main()
