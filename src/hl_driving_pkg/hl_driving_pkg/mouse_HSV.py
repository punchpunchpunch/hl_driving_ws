import cv2
import numpy as np

def show_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 클릭
        pixel = frame[y, x]  # BGR
        hsv_pixel = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"BGR: {pixel}, HSV: {hsv_pixel}")

# 이미지 불러오기
frame = cv2.imread("/home/yeong/Pictures/Screenshots/Screenshot from 2026-02-23 17-03-12.png")

cv2.imshow("Image", frame)
cv2.setMouseCallback("Image", show_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()
