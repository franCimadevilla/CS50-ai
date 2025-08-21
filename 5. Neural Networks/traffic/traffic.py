import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import time

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

DENSE_LAYER_NEURONS = 512
CONV2D_LAYER_NEURONS = 64

class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start
        print(f"⏱️ Epoch {epoch+1} duration: {duration:.2f} seconds")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.train_start
        print(f"✅ Total training time: {total_time:.2f} seconds")


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[TimingCallback()]
    )

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    
    for subfolder in range(NUM_CATEGORIES-1):
        subfolder_path = os.path.join(data_dir, str(subfolder))
        if not os.path.isdir(subfolder_path):
            raise NotADirectoryError()
        
        # Now iterate in EACH FILE NAME of the subfolder path
        for ppm_file_name in os.listdir(subfolder_path):
            ppm_file_path = os.path.join(subfolder_path, str(ppm_file_name))
            
            # Read each corresponding ppm file 
            
            image = cv2.imread(ppm_file_path, cv2.IMREAD_COLOR)
            if image is None:
                raise Exception("PPM file was not read as is invalid: {}".format(ppm_file_path))
            
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            
            images.append(image)
            labels.append(subfolder)
            
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # First use the tf.keras to create a Sequential Network with layers
    model = tf.keras.models.Sequential([
        
        # Convolutional layer of 32 nodes and 3x3 filter kernel each
        tf.keras.layers.Conv2D(
            CONV2D_LAYER_NEURONS, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        
        # Pooling layer to reduce the array dimension, Max-Pooling config
        tf.keras.layers.MaxPooling2D(
            pool_size=(2,2)
        ),
        
        # Flatten units
        tf.keras.layers.Flatten(),
        
        # Adding a hidden dense layer with neurons
        tf.keras.layers.Dense(DENSE_LAYER_NEURONS, activation="relu"),
        
        # Adding dropout index to avoid neuron overfitting over its neighbours
        tf.keras.layers.Dropout(0.5),
        
        # Output layer considering all the output categories
        tf.keras.layers.Dense(NUM_CATEGORIES-1, activation="softmax")
        ])
    
    # compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


if __name__ == "__main__":
    main()
