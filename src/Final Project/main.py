from k_means import KMEANS
import os

# Imports for the model and utilities
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from time import sleep


def join_path(imagepath, file):
    """
    Returns the full path of a file.

    :param imagepath: (str) Path to the folder where the file is
    :param file: (str) File name
    :return: (str) The full path of the file
    """
    return os.path.join(imagepath, file)


def is_file(imagepath, file):
    """
    Returns if the path provided is a file

    :param imagepath: (str) Path to the folder where the file is
    :param file: (str) File name
    :return:  (bool) True if the path is a file, else returns False
    """
    return os.path.isfile(join_path(imagepath, file))


def prediction(model, imagepath):
    """
    Predicts if a leaf image is healthy or not

    :param model: Tensorflow trained model
    :param imagepath: (str) Full path of an image file
    :return: (int) 0 if predicted image is healthy, else returns 1
    """
    class_names = ['diseased', 'healthy']
    img = keras.preprocessing.image.load_img(imagepath, target_size=(512, 512))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    preds = model.predict(img_array)
    score = tf.nn.softmax(preds[0])

    return 0 if class_names[np.argmax(score)] == 'healthy' else 1


def main():
    image_path = 'images'
    if os.path.exists(image_path):
        # Load the predictions model
        model = load_model('Model/model.h5')

        # List all files from a given directory
        filenames = [file for file in os.listdir(image_path) if is_file(image_path, file)]

        # Filter files to only have image files of type JPG, JPEG and PNG
        image_files = [img_file for img_file in filenames if img_file.split('.')[-1] in ['jpg', 'jpeg', 'png']]

        choice_comparison = 'q'
        choice_contour = 'q'

        while choice_comparison not in 'yYnN':
            choice_comparison = str(input('Want to save the comparison plot? (y/n)\n-> ')).strip()
        save_comparison_plot = True if choice_comparison in 'yY' else False

        while choice_contour not in 'yYnN':
            choice_contour = str(input('Want to save the contours plot? (y/n)\n-> ')).strip()
        save_contour_plot = True if choice_contour in 'yY' else False

        for img in image_files:
            full_image_path = os.path.join(image_path, img)
            if prediction(model, full_image_path) == 0:
                print(f'Image {img.split(".")[0]} is a healthy leaf!\n')
                continue
            else:
                print(f'Image {img.split(".")[0]} is a diseased leaf!')
                KMEANS(full_image_path, img, save_comparison_plot, save_contour_plot)
                sleep(0.5)
    else:
        print("Incorrent path")


if __name__ == "__main__":
    main()
