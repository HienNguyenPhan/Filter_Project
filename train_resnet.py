from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from read_data import get_train_data, get_test_data, visualize_points
from skimage.color import gray2rgb
import numpy as np
import tensorflow as tf

imgs_train, points_train = get_train_data()
imgs_train_rgb = np.array([gray2rgb(img.squeeze()) for img in imgs_train])


def get_pretrained_resnet(input_shape=(96, 96, 3), num_classes=30):
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    

    for layer in base_model.layers:
        layer.trainable = False


    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation=None)(x) 

    # Create a new model
    model = Model(inputs=base_model.input, outputs=output)
    return model

def compile_model(model):
    model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

def load_trained_model_resnet():
    model.load_weights('weights/resnet_checkpoint-50.keras')


checkpoint = ModelCheckpoint(filepath='weights/resnet_checkpoint-{epoch:02d}.keras')

if __name__ == "__main__":
    model = get_pretrained_resnet(input_shape=(96, 96, 3), num_classes=30)
    compile_model(model)
    model.fit(
        imgs_train_rgb, points_train,
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint]
    )
