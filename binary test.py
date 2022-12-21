import cv2
import numpy as np


def thres_adj(nn):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary1 = cv2.threshold(gray, nn, 255, cv2.THRESH_BINARY)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    binary = cv2.morphologyEx(binary1, cv2.MORPH_OPEN, se)
    cv2.imshow(mwname, binary)
    edges = cv2.Canny(binary, 50, 100)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=20, minRadius=5, maxRadius=9)
    im1 = np.copy(img)
    if circles is not None:
        for circle in circles[0]:
            x = circle[0]
            y = circle[1]
            r = circle[2]

            cv2.circle(im1, (int(x), int(y)), int(r), (0, 0, 255), 1)
            cv2.circle(im1, (int(x), int(y)), 2, (255, 255, 255), -1)
    cv2.imshow(mwname, binary)
    cv2.imshow(refwname, im1)


if __name__ == "__main__":
    mwname = 'Binary test'
    refwname = 'original'
    cv2.namedWindow(mwname, cv2.WINDOW_AUTOSIZE)

    video_path = './2047.avi'

    capture = cv2.VideoCapture(video_path)
    thresh = 0
    ret, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k_pre, k_curr = -1, -1
    while True:
        # cv2.imshow(mwname, img)
        # cv2.createTrackbar('Threshold', mwname, thresh, 255, thres_adj)
        ret, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
        cv2.putText(binary, f'{thresh}', (5, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), bottomLeftOrigin=False)
        cv2.imshow(mwname, binary)
        keypress = cv2.waitKey(0)
        if keypress == 27:
            break
        if keypress > -1:
            print(keypress)
            k_curr = keypress
        k_pre = keypress
        if k_curr == 3 or k_curr == 100:
            thresh += 1
            if thresh > 255:
                thresh = 255
        elif k_curr == 2 or k_curr == 97:
            thresh -= 1
            if thresh < 0:
                thresh = 0
        k_curr = -1
    # cv2.waitKey(0)
