import cv2
import win32ui
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from hot_calibration import hot_calibration
from numba import jit
import csv
from datetime import datetime


class cvVideo:
    def __init__(self, filename):
        self.video_path = filename
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


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
    global thresh
    thresh = nn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary1 = cv2.threshold(gray, nn, 255, cv2.THRESH_BINARY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    binary = cv2.morphologyEx(binary1, cv2.MORPH_OPEN, se)
    bin_color = cv2.cvtColor(binary1, cv2.COLOR_GRAY2BGR)
    for cr in crop:
        rx1 = cr[0]
        rx2 = cr[1]
        ry1 = cr[2]
        ry2 = cr[3]
        cv2.rectangle(bin_color, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2, 8, 0)

    cv2.imshow(thres_win, bin_color)
    # edges = cv2.Canny(binary, 50, 100)
    # circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=20, minRadius=5, maxRadius=9)
    # im1 = np.copy(img)
    # if circles is not None:
    #     for circle in circles[0]:
    #         x = circle[0]
    #         y = circle[1]
    #         r = circle[2]
    #
    #         cv2.circle(im1, (int(x), int(y)), int(r), (0, 0, 255), 1)
    #         cv2.circle(im1, (int(x), int(y)), 2, (255, 255, 255), -1)
    # cv2.imshow(thres_win, binary)


def blob_use_detector(img_crop, thresh):
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
    params = cv2.SimpleBlobDetector_Params()
    detector = cv2.SimpleBlobDetector.create(params)
    kypts = detector.detect(binary)
    if kypts:
        return kypts[0].pt[0], kypts[0].pt[1]
    else:
        return 0, 0


def blob_use_moments(img_crop, thresh):
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
    binary = cv2.bitwise_not(binary)
    M = cv2.moments(binary)
    return M['m10'] / M['m00'], M['m01'] / M['m00']


def ring_use_numpy(img_crop, thresh):
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    return ring_centroid(binary)


@jit
def ring_centroid(binary):
    ring = np.where(binary != 0)
    x = np.mean(ring[0])
    y = np.mean(ring[1])
    return x, y


if __name__ == '__main__':
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    draw = False
    crop_win = 'Crop Selection'

    dlg = win32ui.CreateFileDialog(True)

    dlg.SetOFNInitialDir('./')
    dlg.DoModal()

    filename = dlg.GetPathName()
    if filename == '':
        print('No File is Selected')
        sys.exit(0)

    cap = cv2.VideoCapture(filename)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video has {total_frame} frames.')
    crop = []
    ret, img = cap.read()
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
    thres_win = 'Threshhold Selection'
    cv2.namedWindow(thres_win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(thres_win, img)
    thresh = 0
    cv2.createTrackbar('Threshold', thres_win, 0, 255, thres_adj)
    cv2.waitKey(0)
    print('Calculating...')
    for cr in crop:
        if cr[0] > cr[1]:
            cr[0], cr[1] = cr[1], cr[0]
        if cr[2] > cr[3]:
            cr[2], cr[3] = cr[3], cr[2]
    # cc_x = np.zeros((len(crop), total_frame), dtype=float)
    # cc_y = np.zeros((len(crop), total_frame), dtype=float)
    cc2_x = np.zeros((len(crop), total_frame), dtype=float)
    cc2_y = np.zeros((len(crop), total_frame), dtype=float)

    t = np.zeros(total_frame, dtype=float)
    l = len(crop)

    for frame in range(total_frame):
        t_frame = cap.get(cv2.CAP_PROP_POS_MSEC)
        t[frame] = t_frame
        for i in range(l):
            img_cr = img[crop[i][2]:crop[i][3], crop[i][0]:crop[i][1]]
            # cc_x[i, frame], cc_y[i, frame] = blob_use_detector(img_cr, thresh)
            # cc2_x[i, frame], cc2_y[i, frame] = blob_use_moments(img_cr, thresh)
            cc2_x[i, frame], cc2_y[i, frame] = ring_use_numpy(img_cr, thresh)
        ret, img = cap.read()
    print('Video analysis finished.')
    csvname = filename + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'

    with open(csvname, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        header = ['particle', 'k_x^EQ', 'k_x^P', 'Err:k_x^P', 'x_eq^P', 'Err:x_eq^P', 'R^2_xP', 'k_x^PA1',
                  'Err:k_x^PA1', 'k_x^PA2', 'Err:k_x^PA2', 'x_eq^PA', 'Err:x_eq^PA', 'R^2_xPA', 'k_y^EQ', 'k_y^P',
                  'Err:k_y^P', 'y_eq^P', 'Err:y_eq^P', 'R^2_yP', 'k_y^PA1', 'Err:k_y^PA1', 'k_y^PA2', 'Err:k_y^PA2',
                  'y_eq^PA', 'Err:y_eq^PA', 'R^2_yPA']
        csvwriter.writerow(header)
        cali = hot_calibration(magEx=True)
        cali_res = np.zeros((len(crop), 3, 2), dtype=float)
        for i in range(l):
            print(f'Particle {i + 1}:')
            print('X:')
            k_x = cali.eq_pa(cc2_x[i, :], t, showplot=True)
            print('Y:')
            k_y = cali.eq_pa(cc2_y[i, :], t, showplot=True)
            tl = [[i + 1], k_x, k_y]
            csvwriter.writerow([item for t in tl for item in t])

    # if l == 2:
    #     ax_prefix = 220
    #     dn = [2, 0]
    #     dxy = 1
    # elif l == 3:
    #     ax_prefix = 320
    #     dn = [2, 2, 0]
    #     dxy = 1
    # elif l == 4:
    #     ax_prefix = 240
    #     dn = [1, 3, 1, 0]
    #     dxy = 2
    # else:
    #     ax_prefix = 120
    #     dn = [0]
    #     dxy = 1
    # fign = 1
    # for i in range(l):
    #     xmean = np.mean(cc2_x[i, :])
    #     xc = (cc2_x[i, :] - xmean) * cali.img_pixel_size
    #     ax = fig.add_subplot(ax_prefix + fign)  # type:axes.Axes
    #     xmin, xmax = np.min(xc), np.max(xc)
    #     dx = (xmax - xmin) / cali.bin_count
    #     x_coords = np.arange(xmin + dx / 2, xmax, dx)
    #     ax.plot(x_coords, cali.gauss_distribution(x_coords, cali_res[i, 0, 0]), 'g', label='Equipartition')
    #     ax.plot(x_coords, cali.gauss_distribution(x_coords, cali_res[i, 1, 0]), 'r', label='Potential Analysis')
    #     ax.plot(x_coords, cali.gauss_distribution(x_coords, cali_res[i, 2, 0]), 'm', label='Potential Analysis alter.')
    #     ax.hist(xc, bins=cali.bin_count, color='C0', density=True, label='X position distribution')
    #     ax.legend(loc='upper right', title=f'kx of particle {i}')
    #     ax = fig.add_subplot(ax_prefix + fign + dxy)
    #     ymean = np.mean(cc2_y[i, :])
    #     yc = (cc2_y[i, :] - ymean) * cali.img_pixel_size
    #     ymin, ymax = np.min(yc), np.max(yc)
    #     dy = (ymax - ymin) / cali.bin_count
    #     y_coords = np.arange(ymin + dy / 2, ymax, dy)
    #     ax.plot(y_coords, cali.gauss_distribution(y_coords, cali_res[i, 0, 1]), 'g', label='Equipartition')
    #     ax.plot(y_coords, cali.gauss_distribution(y_coords, cali_res[i, 1, 1]), 'r', label='Potential Analysis')
    #     ax.plot(y_coords, cali.gauss_distribution(y_coords, cali_res[i, 2, 1]), 'm', label='Potential Analysis alter.')
    #     ax.hist(yc, bins=cali.bin_count, color='C1', density=True, label='y position distribution')
    #     ax.legend(loc='upper right', title=f'ky of particle {i}')

    # plt.show()
