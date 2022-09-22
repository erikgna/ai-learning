import numpy as np

from tensorflow.keras.preprocessing import image
from keras.models import load_model

UPLOAD_FOLDER = 'uploads'

myModel = load_model()

def check_file(filename):
    test_image = image.load_img(UPLOAD_FOLDER+"/"+filename, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    result = myModel.predict(test_image)

    print(result)

check_file('')