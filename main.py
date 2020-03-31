#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:06:08 2020

@author: mac
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
from sklearn.preprocessing import normalize
import pandas as pd
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def create_grid_windows(image, *grid_values):
    # Get image size (128 in our case)
    imgwidth = image.shape[0]
    imgheight = image.shape[0]
    # Initialize the image list that will contain ALL grid images
    grid_images = []
    # For each value n : nxn
    for value in grid_values:
        # Set height and width intervals for the given n : nxn
        height = imgheight // value
        width = imgwidth // value
        for i in range(0,imgheight,height):
            for j in range(0,imgwidth,width):
                grid_images.append(image[j:j+width, i:i+height])
    # After iterating over all grid values, return the list containing ALL grid images
    return grid_images


def HSV_Norm_Hist(grid_windows, numberOfBins):

    hist_list = []
    
    for window in grid_windows:
        # Convert image from RGB to HSV
        hsv_window = cv2.cvtColor(window, cv2.COLOR_BGR2HSV)
        
        # Compute and normalize Hue histogram
        hue = cv2.calcHist([hsv_window[:,:,0]], [0], None, [numberOfBins],[0,256])
        hue = normalize(hue.reshape(1,-1), norm='l1')
        
        # Compute and normalize Saturation histogram
        sat = cv2.calcHist([hsv_window[:,:,1]], [0], None, [numberOfBins],[0,256])
        sat = normalize(sat.reshape(1,-1), norm='l1')
        
        # Compute and normalize Value histogram
        val = cv2.calcHist([hsv_window[:,:,2]], [0], None, [numberOfBins],[0,256])
        val = normalize(val.reshape(1,-1), norm='l1')
        
        # Concatenate three histograms
        hist = np.concatenate((hue,sat,val), axis = 1)
        hist_list.append(hist)
        
    stacked_hist = np.hstack(hist_list)
    
    return stacked_hist



def HOG_Norm_Hist(grid_windows, numberOfBins):
    
    hist_list = []
    
    for window in grid_windows:
        # Convert to grayscale
        gray_window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
        
        # Compute the horizontal and vertical components of the gradients
        gx = cv2.Sobel(gray_window, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray_window, cv2.CV_32F, 0, 1, ksize=1)
        
        # Calculate the gradient orientation angles
        _, angles = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        # Compute and normalize gradient orientation histogram
        hist = cv2.calcHist([angles], [0], None, [numberOfBins],[0,256])
        hist = normalize(hist.reshape(1,-1), norm='l1')
        
        hist_list.append(hist)
        
    stacked_hist = np.hstack(hist_list)
    
    return stacked_hist


directory = "/Users/mac/Desktop/SKU_Recognition_Dataset"

# Get 5 category subfolders list
category_paths = [f.path for f in os.scandir(directory) if f.is_dir()]


# Get 101 product subfolders list
product_paths = []

for path in category_paths:
    product_path = [f.path for f in os.scandir(path) if f.is_dir()]
    for path in product_path:
        product_paths.append(path)

del category_paths, directory, path, product_path


# we will use list throughout the for loops because
# DataFrame append/insert are awful in terms of performance
sku_list = []
hsv_list = []
hog_list = []

# Iterate over 101 product subfolders
for path in product_paths:
    # Get each image for the given product
    image_paths = [f.path for f in os.scandir(path)]
    # find sku by getting the substring after last slash (/)
    sku = re.split("/", path)[-1] 
    # Iterate over each image directory
    for image_path in image_paths[:5]:
        # Create image object and resize it to 128x128
        image = cv2.resize(cv2.imread(image_path), (128,128))
        # Create a list of images based on grid cells (in our case 1x1, 2x2 and 4x4)
        grid_windows = create_grid_windows(image, 1, 2, 4)
        # Obtain histograms for each image
        hsv = HSV_Norm_Hist(grid_windows, argParseNumberOfBins)
        hog = HOG_Norm_Hist(grid_windows, argParseNumberOfBins)
        # Append them into belonging lists
        sku_list.append(sku)
        hsv_list.append(hsv)
        hog_list.append(hog)

del image_path, path, image_paths, product_paths
del grid_windows, image, sku, hog, hsv


# Convert lists to pandas DataFrame
sku_df = pd.DataFrame(sku_list, columns=['sku'])
hsv_df = pd.DataFrame(np.concatenate(hsv_list))
hog_df = pd.DataFrame(np.concatenate(hog_list))

hsv_df = pd.concat([sku_df, hsv_df], axis=1)
hog_df = pd.concat([sku_df, hog_df], axis=1)

del sku_list, hsv_list, hog_list, sku_df


# Split the set into training and subsets by randomly sampling EACH CLASS
train_hsv, test_hsv = train_test_split(hsv_df, test_size=0.2, random_state=42, 
                                     stratify = hsv_df[['sku']])

train_hog, test_hog = train_test_split(hog_df, test_size=0.2, random_state=42, 
                                     stratify = hog_df[['sku']])


# Reset and drop indexes
for df in [train_hsv, test_hsv, train_hog, test_hog]:
    df.reset_index(drop=True, inplace=True)

train_labels = train_hsv['sku'] # hsv/hog does not matter, their order is the same
test_labels = test_hsv['sku'] # hsv/hog does not matter, their order is the same

train_hsv_data = train_hsv.loc[:, train_hsv.columns != 'sku']
test_hsv_data = test_hsv.loc[:, test_hsv.columns != 'sku']
train_hog_data = train_hog.loc[:, train_hog.columns != 'sku']
test_hog_data = test_hog.loc[:, test_hog.columns != 'sku']

del df, hog_df, hsv_df, train_hog, train_hsv, test_hog, test_hsv

# Initialize classifiers
hsv_classifier = KNeighborsClassifier(n_neighbors = argParseNNeighbors)
hog_classifier = KNeighborsClassifier(n_neighbors = argParseNNeighbors)
# Fit classifiers to the training data
hsv_classifier.fit(train_hsv_data, train_labels)
hog_classifier.fit(train_hog_data, train_labels)
# Predict test data
hsv_pred = hsv_classifier.predict(test_hsv_data)
hog_pred = hog_classifier.predict(test_hog_data)

del argParseNNeighbors, argParseNumberOfBins
del test_hog_data, test_hsv_data, train_hog_data, train_hsv_data, train_labels

hsv_acc = accuracy_score(hsv_pred, test_labels)
hog_acc = accuracy_score(hog_pred, test_labels)

del hsv_pred, hog_pred, test_labels

print("HSV Accuracy Score: %.2f" % hsv_acc)
print("HOG Accuracy Score: %.2f" % hog_acc)

