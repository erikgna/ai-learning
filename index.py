from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

IMG_WIDTH, IMG_HEIGHT = 224, 224

TRAIN_DATA_DIR = 'v_data/train'
VALIDATION_DATA_DIR = 'v_data/test'
NB_TRAIN_SAMPLES = 400
NB_VALIDATION_SAMPLES = 100
EPOCHS = 10
BATCH_SIZE = 16

def build_model():
    if K.image_data_format() == 'channels_first':
	    input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
    else:
	    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def train_model(model):
    train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='binary')

    model.fit(
        train_generator,
        steps_per_epoch=NB_TRAIN_SAMPLES // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=NB_VALIDATION_SAMPLES // BATCH_SIZE)


    return model

def save_model(model):
    model.save('saved_model.h5')

def main():
    myModel = build_model()
    myModel = train_model(myModel)
    save_model(myModel)

main()