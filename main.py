import operator

import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import cv2 as cv2
import math
from pprint import pprint

IMAGE1 = 'traffic light.jpg'
IMAGE2 = 'traffic1.jpg'
IMAGE3 = 'traffic2.jpeg'
IMAGE4 = 'traffic3.jpg'


# def find_color(sliding_window, threshold=10):
#     red_level, yellow_level, green_level = 0, 0, 0
#     red_level = np.mean(sliding_window[:, :, 0])
#     green_level = np.mean(sliding_window[:, :, 1])
#     yellow_level = (red_level + green_level) / 2
#     if np.abs(red_level - yellow_level) < threshold:
#         return "red+yellow"
#     if red_level == max(red_level, yellow_level, green_level):
#         return "red"
#     if yellow_level == max(red_level, yellow_level, green_level):
#         return "yellow"
#     else:
#         return "green"


def findColor(sliding_window):
    colors_dict = {}
    # hsv = cv2.cvtColor(sliding_window, cv2.COLOR_BGR2HSV)

    # finding red color
    low_red = np.array([150, 0, 60])
    upper_red = np.array([255, 30, 150])
    mask_red = cv2.inRange(sliding_window, lowerb=low_red, upperb=upper_red)
    colors_dict['red'] = checkGroups(mask_red)

    # finding yellow color
    low_yellow = np.array([220, 90, 0])
    upper_yellow = np.array([255, 255, 60])
    mask_yellow = cv2.inRange(sliding_window, lowerb=low_yellow, upperb=upper_yellow)
    colors_dict['yellow'] = checkGroups(mask_yellow)

    # finding green color
    low_green = np.array([60, 180, 110])
    upper_green = np.array([120, 255, 210])
    mask_green = cv2.inRange(sliding_window, lowerb=low_green, upperb=upper_green)
    colors_dict['green'] = checkGroups(mask_green)

    # merge the masks to get the total result
    total_mask = mask_red + mask_yellow + mask_green

    cv2.imshow('original', sliding_window)
    cv2.imshow('after process', total_mask)

    cv2.waitKey()
    cv2.destroyAllWindows()

    return colors_dict


# Function to detect if the pixels we found is a close group
def checkGroups(mask):
    is_group = False
    line_list = []
    # iterate over the mask matrix
    for index, group in enumerate(mask):
        # checking for each group it have color pixels
        for pixel in group:
            if pixel != 0:
                line_list.append(index)
                break
        continue

    if len(line_list) == 0:
        return is_group
    s = line_list[-1] - line_list[0]

    # check if we have gap between the lines
    if len(line_list) == s + 1:
        is_group = True
    return is_group


def isTrafficLight(sliding_window):
    sliding_window_image = Image.fromarray(np.uint8(sliding_window))
    sliding_window_image = reshape(sliding_window_image)
    plt.imshow(sliding_window_image)
    plt.show()
    sliding_window = img_to_array(sliding_window_image)
    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(sliding_window, axis=0)
    # print('image batch size', image_batch.shape)
    plt.imshow(np.uint8(image_batch[0]))

    # prepare the image for the VGG model
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = vgg_model.predict(processed_image)
    # print predictions

    # convert the probabilities to class labels
    # We will get top 5 predictions which is the default
    label = decode_predictions(predictions)
    # if predictions >= 0.8 && label == :
    #     print('This is my traffic light')
    #     print(label)
    percentage = (label[0][0])[2]
    if label[0][0][1] == 'traffic_light' and percentage >= 0.9:
        print('Accuracy: ', percentage * 100, '%')
        return True
    return False


def reshape(img):  # gets (n, n, 3) numpy image returns (224, 224, 3) numpy image
    basewidth = 224
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return img


def getColors(numpy_image):
    colors = []
    window_size = 150
    original_picture_rows, original_picture_cols = numpy_image.shape[0], numpy_image.shape[1]
    for i in range(0, original_picture_rows - window_size, window_size):
        for j in range(0, original_picture_cols - window_size, window_size):
            sliding_window = numpy_image[i:i + window_size, j:j + window_size, :]
            if isTrafficLight(sliding_window):
                colors.append(findColor(sliding_window))
    return colors


def getFinalColor(colors):
    grades = {
        'red': 0,
        'yellow': 0,
        'green': 0
    }
    for dict in colors:
        if dict['red']:
            grades['red'] += 1
        if dict['yellow']:
            grades['yellow'] += 1
        if dict['green']:
            grades['green'] += 1
    return grades


def getHeighestGrade(grades):
    s_grades = sorted(grades.items(), key=lambda x: x[1])
    if s_grades[-1][1] == s_grades[-2][1]:
        final = s_grades[-2:]
    else:
        final = s_grades[-1]
    return final


def printResults(final_color):
    if type(final_color) is list:
        for res in final_color:
            print('Color: {} >> Grade: {}'.format(res[0], res[1]))
    else:
        print('Color: {} >> Grade: {}'.format(final_color[0], final_color[1]))


# ------------------------  Main  -------------------------- #
# TODO: test another pictures


if __name__ == '__main__':
    # Load the VGG model
    vgg_model = vgg16.VGG16(weights='imagenet')

    # load an image in PIL format
    original = load_img(IMAGE4)

    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)
    plt.imshow(np.uint8(numpy_image))
    plt.show()

    colors = getColors(numpy_image)
    grades = getFinalColor(colors)
    final_color = getHeighestGrade(grades)
    printResults(final_color)

