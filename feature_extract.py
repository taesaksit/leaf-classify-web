import numpy as np
import cv2
from scipy.signal import convolve2d
from sklearn.preprocessing import MinMaxScaler
import warnings
from img2vec_pytorch import Img2Vec
from PIL import Image

warnings.filterwarnings('ignore')


def rgb2gray(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) / 255


def convolution(path):
    images = rgb2gray(path)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image = images
    while True:
        new_image = convolve2d(image, kernel, mode='valid')
        if new_image.size <= 99:
            break
        image = new_image

    final_image = MinMaxScaler().fit_transform(image.reshape(-1, 1))
    return final_image.flatten()


def convolution_from_nn(path):
    image = Image.open(path)
    img2vec = Img2Vec()  
    result = img2vec.get_vec(image)

    return result
