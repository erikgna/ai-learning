from keras.models import load_model
import numpy as np
import tensorflow as tf

def predict(image):
    model = load_model('model.h5')

    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    label = model.predict(img_array)

    score = tf.nn.softmax(label[0])

    prediction_name = np.argmax(score)    

    if prediction_name == 0:
        return 'Caminhão'
    if prediction_name == 1:
        return 'Carro'
    if prediction_name == 2:
        return 'Moto'
    if prediction_name == 3:
        return 'Ônibus'