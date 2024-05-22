import pandas as pd
import numpy as np
import cv2
from scipy.signal import convolve2d
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

def predcit(data , model):

    labels = {
        0:'Black_rot',
        1:'Esca',
        2:'Leaf spot',
        3:'Healthy'
    }

    labels[0]
    data_clone = np.tile(data, (4, 1))
    dis = np.sqrt(np.sum((model - data_clone) ** 2, axis=1))
    return labels[np.argmin(dis)]