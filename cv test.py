import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes


def mouse_draw_rect(event, x, y, flags, param):
    global x1, x2, y1, y2
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y

    if event == cv2.EVENT_LBUTTONUP:
        x2 = x
        y2 = y

        dx = x2 - x1
        dy = y2 - y1

        if dx != 0 and dy != 0:
            icurr[...] = ilast[...]
            cv2.rectangle(icurr, (x1, y1), (x2, y2), (0, 0, 255), 2, 8, 0)
            print('x1:', x1, 'y1:', y1, '->x2:', x2, 'y2', y2)
            ilast[...] = icurr[...]
            crop.append([x1, x2, y1, y2])

        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0

    if event == cv2.EVENT_MOUSEMOVE:
        if x1 == 0 or y1 == 0:
            return
        x2 = x
        y2 = y
        dx = x2 - x1
        dy = y2 - y1

        if dx != 0 and dy != 0:
            icurr[...] = ilast[...]
            cv2.rectangle(icurr, (x1, y1), (x2, y2), (0, 0, 255), 2, 8, 0)


if __name__ == '__main__':
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    crop = []
    video_path = './b2.avi'

    capture = cv2.VideoCapture(video_path)

    fps = capture.get(cv2.CAP_PROP_FPS)

    total_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    t = np.zeros(total_frame, dtype=float)
    cir_c_x = np.zeros(total_frame, dtype=float)
    cir_c_y = np.zeros(total_frame, dtype=float)
    cir_r = np.zeros(total_frame, dtype=float)

    print(fps, total_frame)
    winname = 'image'
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(winname, mouse_draw_rect)
    cropwin = 'crop'
    cv2.namedWindow(cropwin, cv2.WINDOW_AUTOSIZE)

    ret, img = capture.read()
    ilast = np.copy(img)
    icurr = np.copy(img)
    while True:
        cv2.imshow(winname, icurr)
        c = cv2.waitKey(1)
        if c == 27:
            break

    x1 = crop[0][0]
    x2 = crop[0][1]
    y1 = crop[0][2]
    y2 = crop[0][3]
    # cv2.imshow('crop0', img[y1:y2, x1:x2])
    # c = cv2.waitKey(0)
    i = 0
    nn = 0
    print('Hough Circle begins...')
    while True:
        # print(i)
        ki = img[y1:y2, x1:x2]

        gray = cv2.cvtColor(ki, cv2.COLOR_BGR2GRAY)
        vt = capture.get(cv2.CAP_PROP_POS_MSEC)
        t[i] = vt
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.5, 5, param1=100, param2=25, minRadius=5, maxRadius=9)
        if circles is not None:
            nn += 1
            # print(circles[0])
            x = circles[0][0][0]
            y = circles[0][0][1]
            r = circles[0][0][2]
            cir_c_x[i] = x
            cir_c_y[i] = y
            cir_r[i] = r
            cv2.circle(ki, (int(x), int(y)), int(r), (0, 0, 255), 1, 10, 0)
            # cv2.imshow(cropwin, ki)
            # print(vt, x, y, r, len(circles))
            # cv2.waitKey(0)
        i += 1

        ret, img = capture.read()
        if not ret:
            break
    print(nn, total_frame)

    cv2.destroyAllWindows()
    cc_x_mean = np.mean(cir_c_x)
    cc_y_mean = np.mean(cir_c_y)
    cc_r_mean = np.mean(cir_r)
    print(np.std(cir_r))

    ax1 = plt.subplot(311)  # type:axes.Axes
    ax1.set_ylim(cc_x_mean * 0.9, cc_x_mean * 1.1)

    plt.plot(t, cir_c_x)
    ax2 = plt.subplot(312)
    ax2.set_ylim(cc_y_mean * 0.9, cc_y_mean * 1.1)
    plt.plot(t, cir_c_y)
    ax3 = plt.subplot(313)
    ax3.set_ylim(cc_r_mean * 0.85, cc_r_mean * 1.15)
    plt.plot(t, cir_r)

    plt.show()
