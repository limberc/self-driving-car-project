#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) 2017 - Limber Cheng <cheng@limberence.com> 
# @Time : 15/03/2018 22:58
# @Author : Limber Cheng
# @File : load_data
# @Software: PyCharm
from glob import glob


def load_images():
    '''
    Note: You should first get and unzip the data then to use this function to process the data.
    Download the data here:
    These data is provided by Udacity.
    [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
    [not-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)
    :return: [list] Cars and Notcars images address
    '''
    images_nv = glob('non-vehicles/*/*')
    images_v = glob('vehicles/*/*')
    cars = []
    notcars = []
    for image in images_nv:
        notcars.append(image)
    for image in images_v:
        cars.append(image)
    return cars, notcars
