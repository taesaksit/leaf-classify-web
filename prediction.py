import numpy as np
import pandas as pd

modelDataFrame = pd.read_excel('/Users/tae/Desktop/WOA-Project @2025/model/model_dataset512.xlsx' ,header=None )
model = modelDataFrame.to_numpy()

import warnings
warnings.filterwarnings('ignore')

def predcit(data):

    labels = {
        0:'Black_rot',
        1:'Esca',
        2:'Leaf spot',
        3:'Healthy'
    }

    cloneData = np.tile(data, (4, 1))
    distance = np.sqrt(np.sum((model - cloneData) ** 2, axis=1))
    return labels[np.argmin(distance)]