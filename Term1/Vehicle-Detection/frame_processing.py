#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (c) 2017 - Limber Cheng <cheng@limberence.com> 
# @Time : 15/03/2018 23:37
# @Author : Limber Cheng
# @File : frame_processing
# @Software: PyCharm
import numpy as np
from scipy.ndimage.measurements import label


def add_heat(heatmap, bbox_list):
    '''
    :return: updated heatmap
    '''
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    '''
    Zero out pixels below the threshold in the heatmap
    :param heatmap:
    :param threshold:
    :return: heatmap
    '''
    heatmap[heatmap < threshold] = 0
    return heatmap


def filt(a, b, alpha):
    '''
    Smooth the car boxes
    :param a:
    :param b:
    :param alpha:
    :return:
    '''
    return a * alpha + (1.0 - alpha) * b


def len_points(p1, p2):  # Distance beetween two points
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def track_to_box(p):
    '''
    Create box coordinates out of its center and span
    :param p:
    :return:
    '''
    return ((int(p[0] - p[2]), int(p[1] - p[3])), (int(p[0] + p[2]), int(p[1] + p[3])))


def draw_labeled_bboxes(labels):
    global track_list
    track_list_l = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # img = draw_boxes(np.copy(img), [bbox], color=(255,0,255), thick=3)
        size_x = (bbox[1][0] - bbox[0][0]) / 2.0  # Size of the found box
        size_y = (bbox[1][1] - bbox[0][1]) / 2.0
        asp_d = size_x / size_y
        size_m = (size_x + size_y) / 2
        x = size_x + bbox[0][0]
        y = size_y + bbox[0][1]
        asp = (y - Y_MIN) / 30.0 + 1.2  # Best rectangle aspect ratio for the box (coefficients from perspectieve measurements and experiments)
        if x > 1050 or x < 230:
            asp *= 1.4
        asp = max(asp, asp_d)  # for several cars chunk
        size_ya = np.sqrt(size_x * size_y / asp)
        size_xa = int(size_ya * asp)
        size_ya = int(size_ya)
        if x > (-3.049 * y + 1809):  # If the rectangle on the road, coordinates estimated from a test image
            track_list_l.append(np.array([x, y, size_xa, size_ya]))
            if len(track_list) > 0:
                track_l = track_list_l[-1]
                dist = []
                for track in track_list:
                    dist.append(len_points(track, track_l))
                min_d = min(dist)
                if min_d < THRES_LEN:
                    ind = dist.index(min_d)
                    track_list_l[-1] = filt(track_list[ind], track_list_l[-1], ALPHA)
    track_list = track_list_l
    boxes = []
    for track in track_list_l:
        boxes.append(track_to_box(track))
    return boxes


def frame_proc(img, lane=False, video=False, vis=False):
    if (video and n_count % 2 == 0) or not video:  # Skip every second video frame
        global heat_p, boxes_p, n_count
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        boxes = []
        boxes = find_cars(img, 400, 650, 950, 1280, 2.0, 2)
        boxes += find_cars(img, 400, 500, 950, 1280, 1.5, 2)
        boxes += find_cars(img, 400, 650, 0, 330, 2.0, 2)
        boxes += find_cars(img, 400, 500, 0, 330, 1.5, 2)
        boxes += find_cars(img, 400, 460, 330, 950, 0.75, 3)
        for track in track_list:
            y_loc = track[1] + track[3]
            lane_w = (y_loc * 2.841 - 1170.0) / 3.0
            if lane_w < 96:
                lane_w = 96
            lane_h = lane_w / 1.2
            lane_w = max(lane_w, track[2])
            xs = track[0] - lane_w
            xf = track[0] + lane_w
            if track[1] < Y_MIN:
                track[1] = Y_MIN
            ys = track[1] - lane_h
            yf = track[1] + lane_h
            if xs < 0: xs = 0
            if xf > 1280: xf = 1280
            if ys < Y_MIN - 40: ys = Y_MIN - 40
            if yf > 720: yf = 720
            size_sq = lane_w / (0.015 * lane_w + 0.3)
            scale = size_sq / 64.0
            # Apply multi scale image windows
            boxes += find_cars(img, ys, yf, xs, xf, scale, 2)
            boxes += find_cars(img, ys, yf, xs, xf, scale * 1.25, 2)
            boxes += find_cars(img, ys, yf, xs, xf, scale * 1.5, 2)
            boxes += find_cars(img, ys, yf, xs, xf, scale * 1.75, 2)
            if vis:
                cv2.rectangle(img, (int(xs), int(ys)), (int(xf), int(yf)), color=(0, 255, 0), thickness=3)
        heat = add_heat(heat, boxes)
        heat_l = heat_p + heat
        heat_p = heat
        heat_l = apply_threshold(heat_l, THRES)  # Apply threshold to help remove false positives
        # Visualize the heatmap when displaying
        heatmap = np.clip(heat_l, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        # print((labels[0]))
        cars_boxes = draw_labeled_bboxes(labels)
        boxes_p = cars_boxes

    else:
        cars_boxes = boxes_p
    if lane:  # If we was asked to draw the lane line, do it
        if video:
            img = laneline.draw_lane(img, True)
        else:
            img = laneline.draw_lane(img, False)
    imp = draw_boxes(np.copy(img), cars_boxes, color=(0, 0, 255), thick=6)
    if vis:
        imp = draw_boxes(imp, boxes, color=(0, 255, 255), thick=2)
        for track in track_list:
            cv2.circle(imp, (int(track[0]), int(track[1])), 5, color=(255, 0, 255), thickness=4)
    n_count += 1
    return imp
