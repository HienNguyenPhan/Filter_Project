from train import load_trained_model, get_model
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

imgs_test = get_test_data()

def test_model(model):    
    data_path = join('','*g')
    files = glob.glob(data_path)
    for i,f1 in enumerate(files):       
        if f1 == 'Capture_2.jpg':
            img = imread(f1)
            img = rgb2gray(img)    
            test_img = resize(img, (96,96))    
    test_img = np.array(test_img)
    test_img_input = np.reshape(test_img, (1,96,96,1))  
    prediction = model.predict(test_img_input)     
    visualize_points(test_img, prediction[0])
    

    for i in range(len(imgs_test)):
        test_img_input = np.reshape(imgs_test[i], (1,96,96,1))   
        prediction = model.predict(test_img_input)    
        visualize_points(imgs_test[i], prediction[0])
        if i == 0:
            break

model = get_model()
load_trained_model(model)
test_model(model)