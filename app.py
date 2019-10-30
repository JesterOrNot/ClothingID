from model import ClothingID
import numpy as np
predictions = ClothingID()
print(np.argmax(predictions[0]))
