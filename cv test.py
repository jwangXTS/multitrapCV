import cv2
import numpy as np


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
    video_path = './bright1.avi'

    capture = cv2.VideoCapture(video_path)

    fps = capture.get(cv2.CAP_PROP_FPS)

    total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    print(fps, total_frame)
    winname = 'image'
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(winname, mouse_draw_rect)

    ret, img = capture.read()
    ilast = np.copy(img)
    icurr = np.copy(img)
    while True:
        cv2.imshow(winname, icurr)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cv2.destroyAllWindows()
