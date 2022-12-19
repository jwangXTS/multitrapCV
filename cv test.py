import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from hot_calibration import hot_calibration


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
            ilast[...] = icurr[...]
            crop.append([x1, x2, y1, y2])

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


if __name__ == '__main__':
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    crop = []
    video_path = './F=27.55.avi'
    draw = False

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
    binwin = 'binary'
    cv2.namedWindow(binwin, cv2.WINDOW_AUTOSIZE)

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
    print('Blob Circle begins...')
    draw = False
    while True:
        # print(i)
        ki = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(ki, cv2.COLOR_BGR2GRAY)
        vt = capture.get(cv2.CAP_PROP_POS_MSEC)
        t[i] = vt
        ret, binary = cv2.threshold(gray, 54, 255, cv2.THRESH_BINARY)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
        # edges = cv2.Canny(binary, 50, 100)
        # circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 5, param1=100, param2=10, minRadius=8, maxRadius=8)
        # if circles is not None:
        #     nn += 1
        #     # print(circles[0])
        #     xx = circles[0][0][0]
        #     yy = circles[0][0][1]
        #     rr = circles[0][0][2]
        #     cir_c_x[i] = xx
        #     cir_c_y[i] = yy
        #     cir_r[i] = rr
        #     cv2.circle(ki, (int(xx), int(yy)), int(rr), (0, 0, 255), 1, 10, 0)
        #     # cv2.imshow(cropwin, ki)
        #     # print(vt, xx, yy, rr, len(circles))
        #     # cv2.waitKey(0)
        params = cv2.SimpleBlobDetector_Params()
        # params.blobColor = 255
        detector = cv2.SimpleBlobDetector.create(params)
        kypts = detector.detect(binary)
        for kp in kypts:
            nn += 1
            xx = kp.pt[0]
            yy = kp.pt[1]
            rr = kp.size / 2
            cir_c_x[i] = xx
            cir_c_y[i] = yy
            cir_r[i] = rr
            if rr > 15:
                cv2.circle(ki, (int(xx), int(yy)), int(rr), (0, 0, 255), 1, 10, 0)
                cv2.imshow(cropwin, ki)
                cv2.imshow(binwin, binary)
                print(vt, xx, yy, rr)
                cv2.waitKey(0)
        i += 1

        ret, img = capture.read()
        if not ret:
            break
    print(nn, total_frame)

    nzero = np.where(cir_r != 0)
    cc_x_res = cir_c_x[nzero]
    cc_y_res = cir_c_y[nzero]
    cc_r_res = cir_r[nzero]
    x_histo = np.histogram(cc_x_res, bins=50, range=(np.min(cc_x_res), np.max(cc_x_res)))
    cv2.destroyAllWindows()
    cc_x_mean = np.mean(cc_x_res)
    cc_y_mean = np.mean(cc_y_res)
    cc_r_mean = np.mean(cc_r_res)
    tt = t[nzero]
    print(np.std(cc_r_res))
    fig = plt.figure(figsize=(12, 9))  # type:figure.Figure

    ax1 = fig.add_subplot(321)  # type:axes.Axes
    # ax1.set_ylim(cc_x_mean * 0.9, cc_x_mean * 1.1)
    plt.plot(tt, cc_x_res)
    ax1h = fig.add_subplot(322)
    plt.hist(cc_x_res, orientation='horizontal', bins=25, density=True)

    ax2 = fig.add_subplot(323)
    # ax2.set_ylim(cc_y_mean * 0.9, cc_y_mean * 1.1)
    plt.plot(tt, cc_y_res)
    ax2h = fig.add_subplot(324)
    plt.hist(cc_y_res, orientation='horizontal', bins=25, density=True)
    ax3 = fig.add_subplot(325)
    # ax3.set_ylim(cc_r_mean * 0.85, cc_r_mean * 1.15)
    plt.plot(tt, cc_r_res)
    ax3h = fig.add_subplot(326)
    plt.hist(cc_r_res, orientation='horizontal', bins=25, density=True)

    plt.show()

    cali = hot_calibration(magEx=True)
    print(f'Equipartition: kx={cali.equipartition(cc_x_res)}, ky={cali.equipartition(cc_y_res)}')
    pkx = cali.potential_analysis(cc_x_res, showplot=True)
    pky = cali.potential_analysis(cc_y_res, showplot=True)
    print(f'Potential analysis: kx={pkx}, ky={pky}')
