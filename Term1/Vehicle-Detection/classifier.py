#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) 2017 - Limber Cheng <cheng@limberence.com> 
# @Author : Limber Cheng
# @File : classifier
# @Software: PyCharm
import pickle
import time
from os.path import exists

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from feature_extraction import extract_features
from load_data import load_images


class Classifier:
    """
    Define parameters for feature extraction
    :param color_space: Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    :param orient: HOG orientations
    :param pix_per_cell: HOG pixels per cell
    :param cell_per_block: HOG cells per block
    :param hog_channel: Can be 0, 1, 2, or "ALL"
    :param pix_per_cell: HOG pixels per cell
    :param spatial_size: Spatial binning dimensions
    :param spatial_feat: Spatial features on or off
    :param hist_bins: Number of histogram bins
    :param hist_feat: Histogram features on or off
    :param hog_feat: HOG features on or off
    :return: Classifier Using SVC.
    """

    def __init__(self, color_space='LUV',
                 orient=8,
                 pix_per_cell=8,
                 cell_per_block=2,
                 hog_channel=0,
                 spatial_size=(16, 16),
                 spatial_feat=True,
                 hist_bins=32,
                 hist_feat=True,
                 hog_feat=True):
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.spatial_feat = spatial_feat
        self.hist_bins = hist_bins
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.car_features())
        if exists('saved-model/saved-svc.pickle'):
            self.have_model = True

    def get_features(self, data):
        return extract_features(data, color_space=self.color_space,
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                orient=self.orient, pix_per_cell=self.pix_per_cell,
                                cell_per_block=self.cell_per_block,
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)

    def car_features(self):
        """
        Process the two features
        :return: a tuple of two features(car and nocar)
        """
        cars, no_cars = load_images()
        car_features = self.get_features(cars)
        nocar_features = self.get_features(no_cars)
        return (car_features, nocar_features)

    def split_data(self, features):
        '''
        :param features: A Tuple of features
        :return: Precessed Data
        '''
        X = np.vstack(tup=features).astype(np.float64)
        X_scaler = StandardScaler().fit(X).transform(X)  # Fit a per-column scaler and apply the scaler to X
        y_tup = (np.ones(features[0]), np.zeros(features[1]))
        y = np.hstack(tup=y_tup)  # Define the labels vector
        # Split up data into randomized training and test sets
        return train_test_split(scaled_X, y, test_size=0.2, random_state=22)

    def model(self):
        if self.have_model:
            print("Loading the SVC model")
            with open('saved-model/saved-svc.pickle', 'rb') as f:
                svc = pickle.load(f)
        elif not self.have_model:
            # print("Training the SVC model")
            # features = self.car_features()
            # print('Car samples: ', len(features[0]))
            # print('Notcar samples: ', len(features[1]))
            # print('Using:', orient, 'orientations', pix_per_cell,
            #       'pixels per cell and', cell_per_block, 'cells per block')
            # print('Feature vector length:', len(X_train[0]))
            svc = LinearSVC(loss='hinge')  # Use a linear SVC
            # X_train, X_test, y_train, y_test = self.split_data((car_features, notcar_features))
            svc.fit(self.X_train, self.y_train)  # Train the classifier
            print("Saving the Model")
            with open('saved-model/saved-svc.pickle', 'wb') as f:
                pickle.dump(svc, f)
        return svc

    def test_model(self):
        """
        Test the Model Accurancy
        :param model: Classifier
        :return: Model point
        """
        t = time.time()
        model = self.model()
        print('Test Accuracy of SVC = ', round(model.score(self.X_test, self.y_test), 4))  # Check the score of the SVC
        t2 = time.time()
        print(round(t2 - t, 2), 'seconds to test')
