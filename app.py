from model import ClothingID
import numpy as np

def main():
    predictions = ClothingID()
    response = np.argmax(predictions[0])
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