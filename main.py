import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import cv2 as cv2

IMAGE2 = 'traffic light.jpg'
IMAGE1 = 'traffic1.jpg'

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

def find_color(sliding_window):
    # hsv = cv2.cvtColor(sliding_window, cv2.COLOR_BGR2HSV)

    # finding red color
    low_red=np.array([150,0,60])
    upper_red = np.array([255,50,150])
    mask_red = cv2.inRange(sliding_window, lowerb=low_red, upperb=upper_red)

    # finding yellow color
    low_yellow = np.array([220, 90, 0])
    upper_yello = np.array([255, 255, 60])
    mask_yellow = cv2.inRange(sliding_window, lowerb=low_yellow, upperb=upper_yello)

    # merge the masks to get the total resualt
    total_mask = mask_red + mask_yellow

    cv2.imshow('original', sliding_window)
    cv2.imshow('after process', total_mask)


    cv2.waitKey()
    cv2.destroyAllWindows()
    return True

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
    if label[0][0][1] == 'traffic_light': #top label
        print(label[0][0])
        return True
    return False


def reshape(img): # gets (n, n, 3) numpy image returns (224, 224, 3) numpy image
    basewidth = 224
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return img

#reshape()
#Load the VGG model
vgg_model = vgg16.VGG16(weights='imagenet')




# filename = 'traffic light.jpg'
# load an image in PIL format
original = load_img(IMAGE1)
print('PIL image size',original.size)
plt.imshow(original)
plt.show()

# convert the PIL image to a numpy array
# IN PIL - image is in (width, height, channel)
# In Numpy - image is in (height, width, channel)
numpy_image = img_to_array(original)
plt.imshow(np.uint8(numpy_image))
plt.show()
window_size=150

original_picture_rows, original_picture_cols = numpy_image.shape[0], numpy_image.shape[1]
for i in range(0, original_picture_rows-window_size, window_size):
    for j in range(0, original_picture_cols-window_size, window_size):
        sliding_window = numpy_image[i:i+window_size, j:j+window_size, :]
        if(isTrafficLight(sliding_window) == True):
            print(find_color(sliding_window))


