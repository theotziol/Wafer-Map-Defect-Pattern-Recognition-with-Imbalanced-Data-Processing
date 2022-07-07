import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Input, Add, Dense, Flatten, MaxPooling2D, BatchNormalization, Dropout, Convolution2DTranspose


def new_model():
    
    model = Sequential()
    shape = (128,128,1)
    filters = 16
    model.add(Conv2D(filters, (3,3), padding = 'same', activation='relu', input_shape =shape ))
    model.add(Conv2D(2*filters, (3,3), padding = 'same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(2*filters, (3,3), padding = 'same', activation='relu'))
    model.add(Conv2D(4*filters, (3,3), padding = 'same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(8*filters, (3,3), padding = 'same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(16*filters, (3,3), padding = 'same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(16*filters, (3,3), padding = 'same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(16*filters, (1,1), padding = 'same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    #model.add(BatchNormalization())
    model.add(Dense(128,activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation= 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation= 'relu'))
    model.add(Dense(9,activation= 'softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy',keras.metrics.Precision(),tf.keras.metrics.Recall()] )
    
    return model