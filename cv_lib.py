import cv2
import numpy as np


class cv_lib:
    single_crop = False

    def __init__(self):
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0
        self.draw = False
        self.inverted = False
        self.crop = []
        self.iall = []
        self.ilast = []
        self.icurr = []
        self.single_crop = False
        self.img = None

    def mouse_draw_rect(self, event, x, y, flags, param):
        # global x1, x2, y1, y2, draw, icurr, ilast, iall

        if event == cv2.EVENT_LBUTTONDOWN:
            self.x1 = x
            self.y1 = y
            self.draw = True

        if event == cv2.EVENT_LBUTTONUP:
            self.x2 = x
            self.y2 = y

            dx = self.x2 - self.x1
            dy = self.y2 - self.y1

            if dx != 0 and dy != 0:
                self.icurr[...] = self.ilast[...]
                cv2.rectangle(self.icurr, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255), 2, 8, 0)
                print('x1:', self.x1, 'y1:', self.y1, '->x2:', self.x2, 'y2', self.y2)
                if self.single_crop:
                    self.crop = [[self.x1, self.x2, self.y1, self.y2]]
                    self.iall = [np.copy(self.ilast)]
                else:
                    self.iall.append(np.copy(self.ilast))
                    self.ilast[...] = self.icurr[...]
                    self.crop.append([self.x1, self.x2, self.y1, self.y2])
                # print(len(iall))
            self.x1 = 0
            self.x2 = 0
            self.y1 = 0
            self.y2 = 0
            self.draw = False

        if event == cv2.EVENT_MOUSEMOVE:
            if not self.draw:
                return
            if self.x1 == 0 or self.y1 == 0:
                return
            self.x2 = x
            self.y2 = y
            dx = self.x2 - self.x1
            dy = self.y2 - self.y1

            if dx != 0 and dy != 0:
                self.icurr[...] = self.ilast[...]
                cv2.rectangle(self.icurr, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255), 2, 8, 0)

    def remove_click(self):
        # print(len(iall))
        if len(self.iall) > 0:
            self.icurr[...] = self.iall[-1][...]
            self.ilast[...] = self.iall[-1][...]
            self.iall.pop()
            self.crop.pop()
        else:
            self.img_init(self.img)
            self.iall = []
            self.crop = []

    def img_init(self, img: np.ndarray):
        self.icurr = np.copy(img)
        self.ilast = np.copy(img)


cvlib = cv_lib()
cvlib.single_crop = True
cv2.namedWindow('123', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('123', cvlib.mouse_draw_rect)

img = cv2.imread('./image_20230613-161709.tiff')

cvlib.icurr = np.copy(img)
cvlib.ilast = np.copy(img)

while True:
    cv2.imshow('123', cvlib.icurr)
    c = cv2.waitKey(1)
    if c == 13:
        break
print(cvlib.crop)
