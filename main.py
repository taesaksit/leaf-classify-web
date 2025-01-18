# import numpy as np
# import pandas as pd
# from feature_extract import convolution , convolution_from_nn
# from prediction import predcit
# from watershed import detection

# # Load models with correct dimensions
# model_100 = pd.read_excel('/Users/tae/Desktop/Project/Dataset/modeldataset100.xlsx', index_col=None, header=None)
# model_512 = pd.read_excel('/Users/tae/Desktop/Project/Dataset/modeldataset512.xlsx', index_col=None, header=None)

# def predict512(path):
#     image_detected = None
#     dataset = convolution_from_nn(path)
#     predicted = predcit(dataset, model_512)
#     if predicted != 'Healthy':
#         image_detected = detection(path)
#     return predicted, image_detected

# def predict100(path):
#     image_detected = None
#     dataset = convolution(path)
#     predicted = predcit(dataset, model_100)
#     if predicted != 'Healthy':
#         image_detected = detection(path)
#     return predicted, image_detected
