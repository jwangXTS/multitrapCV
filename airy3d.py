import cv2
import win32ui, win32gui, win32con
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from hot_calibration import hot_calibration

import csv
from datetime import datetime
import pywintypes
import os
from time import time


def mouse_draw_rect(event, x, y, flags, param):
    global x1, x2, y1, y2, draw

    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
        draw = True

    if event == cv2.EVENT_LBUTTONUP:
        x2 = x
        y2 = y

        dx = x2 - x1
        dy = y2 - y1

        if dx != 0 and dy != 0:
            icurr[...] = ilast[...]
            cv2.rectangle(icurr, (x1, y1), (x2, y2), (0, 0, 255), 2, 8, 0)
            print('x1:', x1, 'y1:', y1, '->x2:', x2, 'y2', y2)
            iall.append(np.copy(ilast))
            ilast[...] = icurr[...]
            crop.append([x1, x2, y1, y2])
            # print(len(iall))
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0
        draw = False

    if event == cv2.EVENT_MOUSEMOVE:
        if not draw:
            return
        if x1 == 0 or y1 == 0:
            return
        x2 = x
        y2 = y
        dx = x2 - x1
        dy = y2 - y1

        if dx != 0 and dy != 0:
            icurr[...] = ilast[...]
            cv2.rectangle(icurr, (x1, y1), (x2, y2), (0, 0, 255), 2, 8, 0)


def remove_click():
    global icurr, ilast, img, crop, iall
    print(len(iall))
    if len(iall) > 0:
        icurr[...] = iall[-1][...]
        ilast[...] = iall[-1][...]
        iall.pop()
        crop.pop()
    else:
        icurr = np.copy(img)
        ilast = np.copy(img)
        iall = []
        crop = []


def thres_adj(nn):
    global thresh, inverted
    thresh = nn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary1 = cv2.threshold(gray, nn, 255, cv2.THRESH_BINARY)
    if inverted:
        binary1 = cv2.bitwise_not(binary1)
    # se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    # binary = cv2.morphologyEx(binary1, cv2.MORPH_OPEN, se)
    bin_color = cv2.cvtColor(binary1, cv2.COLOR_GRAY2BGR)
    for cr in crop:
        rx1 = cr[0]
        rx2 = cr[1]
        ry1 = cr[2]
        ry2 = cr[3]
        cv2.rectangle(bin_color, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2, 8, 0)

    cv2.imshow(thres_win, bin_color)


if __name__ == '__main__':
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    draw = False
    crop_win = 'Crop Selection'
    inverted = False

    try:
        ret = win32gui.GetOpenFileNameW(None,
                                        Flags=win32con.OFN_ALLOWMULTISELECT | win32con.OFN_FILEMUSTEXIST | win32con.OFN_HIDEREADONLY | win32con.OFN_EXPLORER,
                                        Title='Select File(s)')
    except pywintypes.error as e:
        if e.winerror == 0:
            print('Cancelled')
        else:
            print('Misc errors')
        sys.exit(0)

    fsplit = ret[0].split('\x00')
    print(fsplit)
    if len(fsplit) == 1:
        filenames = fsplit
    else:
        filenames = []
        dirname = fsplit[0]
        for filename in fsplit[1:]:
            filenames.append(os.path.join(dirname, filename))

    img = cv2.imread(filenames[0], cv2.IMREAD_COLOR)


    # print(f'Video has {total_frame} frames.')
    crop = []

    iall = []
    ilast = np.copy(img)
    icurr = np.copy(img)
    cv2.namedWindow(crop_win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(crop_win, mouse_draw_rect)
    pre_c = -1
    cur_c = -1
    while True:
        cv2.imshow(crop_win, icurr)
        # if len(iall)>0:
        #     cv2.imshow('123',iall[-1])
        c = cv2.waitKey(1)
        if c == 13:
            break
        if c > -1 and c != pre_c:
            cur_c = c
        pre_c = c
        if cur_c == 46 or cur_c == 8:
            remove_click()
            cur_c = -1
    # print(filename)
    cv2.destroyWindow(crop_win)
    thres_win = 'Threshold Selection'
    cv2.namedWindow(thres_win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(thres_win, img)
    thresh = 0
    cv2.createTrackbar('Threshold', thres_win, 0, 255, thres_adj)
    while True:
        c = cv2.waitKey(0)
        if c == 13:
            break
        if c == 73 or c == 105:
            inverted = not inverted
            thres_adj(thresh)
    # print('Calculating...')
