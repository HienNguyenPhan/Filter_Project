import numpy as np
from read_data import get_train_data, get_test_data, visualize_points
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, Input
)
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from skimage.io import imshow
from os.path import join
import glob
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

imgs_train, points_train = get_train_data()
imgs_test = get_test_data()

def get_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=(96,96,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(64, kernel_size=1, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(30))
    return model

def compile_model(model):      
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])

def train_model(model):         
    # checkpoint = ModelCheckpoint(filepath='weights/checkpoint-{epoch:02d}.keras')
    model.fit(imgs_train, points_train, epochs=300, batch_size=100)

def load_trained_model(model):
    model.load_weights('weights/checkpoint-300.keras')

if __name__ == "__main__":
    model = get_model()
    compile_model(model)
    train_model(model)