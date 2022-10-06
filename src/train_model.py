import tensorflow as tf
import tensorflowjs as tfjs

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

IMG_WIDTH, IMG_HEIGHT = 224, 224
TRAIN_DATA_DIR = 'src/page/uploads'
EPOCHS = 15
BATCH_SIZE = 32

def train_model():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    class_names = train_ds.class_names
    
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=(IMG_HEIGHT,
                                        IMG_WIDTH,
                                        3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    num_classes = len(class_names)

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.fit(        
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds)    

    return model

def save_model(model):
    model.save('model.h5')
    tfjs.converters.save_keras_model(model,'js-model')

def main():
    myModel = train_model()
    save_model(myModel)

main()