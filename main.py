import cv2 
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
model = load_model("./model.h5")
image = cv2.imread("image(26).jpg") 
new_image = cv2.resize(image,(412,412)) 
print(np.argmax(model.predict(np.array([new_image]))))

             


             
             
             