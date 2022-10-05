from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

def predict(path):
    model = load_model('saved_model.h5')

    image = load_img(path, target_size=(224, 224))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1,224,224,3)
    label = model.predict(img)
    if(label[0][0] > 0.6):
        print("Plane")
    else:
        print("Car")
