import numpy as np
import cv2
img_dir = './testdata/1.jpg'

def onChange(pos):  # 트랙바 핸들러
    global img
    # 트랙바의 값 받아오기
    # 트랙바 이름, 윈도우 창 이름
    thresh = cv2.getTrackbarPos('threshold', 'Trackbar Windows')
    maxval = cv2.getTrackbarPos("maxValue", "Trackbar Windows")
    _, binary = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)

    cv2.imshow('Trackbar Windows', binary)


src = cv2.imread(img_dir,  cv2.IMREAD_GRAYSCALE)
# cv2.imshow('img', img)  # GUI(윈도우)창 생성, 이미지 보여주기

# 트랙바 생성
# 트랙바 이름, 윈도우 창 이름, 최소값, 최대값, 콜백 함수
cv2.namedWindow("Trackbar Windows", flags=cv2.WINDOW_KEEPRATIO )

cv2.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange)
cv2.createTrackbar("maxValue", "Trackbar Windows", 0, 255, lambda x : x)

cv2.setTrackbarPos("threshold", "Trackbar Windows", 127)
cv2.setTrackbarPos("maxValue", "Trackbar Windows", 255)


cv2.waitKey()
cv2.destroyAllWindows()
