import win32ui
import cv2
import sys
import struct

dlg = win32ui.CreateFileDialog(True)

dlg.SetOFNInitialDir('./')
dlg.DoModal()

filename = dlg.GetPathName()
if filename == '':
    print('No File is Selected')
    sys.exit(0)

cap = cv2.VideoCapture(filename)
cap_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
# print(struct.pack('<I',cap_fourcc))
# sys.exit(0)
fps = cap.get(cv2.CAP_PROP_FPS)
framew = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'RGBA')

output_path = filename + '_crop.avi'
out = cv2.VideoWriter(output_path, cap_fourcc, fps, (framew, frameh))
while True:
    ret, frame = cap.read()
    if not ret:
        break
    t = cap.get(cv2.CAP_PROP_POS_MSEC)
    if 19000 < t < 19300:
        pass
    else:
        out.write(frame)
