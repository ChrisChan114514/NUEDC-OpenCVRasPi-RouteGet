import cv2
import numpy as np


# 初始化阈值
red_lower_1 = np.array([0, 43, 46])
red_upper_1 = np.array([10, 255, 255])
red_lower_2 = np.array([156, 43, 46])
red_upper_2 = np.array([180, 255, 255])


def nothing(x):
    pass


# 创建窗口
cv2.namedWindow('Trackbars')

# 创建滑动条，用于调整红色的HSV上下限
cv2.createTrackbar('H_low1', 'Trackbars', red_lower_1[0], 180, nothing)
cv2.createTrackbar('S_low1', 'Trackbars', red_lower_1[1], 255, nothing)
cv2.createTrackbar('V_low1', 'Trackbars', red_lower_1[2], 255, nothing)
cv2.createTrackbar('H_high1', 'Trackbars', red_upper_1[0], 180, nothing)
cv2.createTrackbar('S_high1', 'Trackbars', red_upper_1[1], 255, nothing)
cv2.createTrackbar('V_high1', 'Trackbars', red_upper_1[2], 255, nothing)
cv2.createTrackbar('H_low2', 'Trackbars', red_lower_2[0], 180, nothing)
cv2.createTrackbar('S_low2', 'Trackbars', red_lower_2[1], 255, nothing)
cv2.createTrackbar('V_low2', 'Trackbars', red_lower_2[2], 255, nothing)
cv2.createTrackbar('H_high2', 'Trackbars', red_upper_2[0], 180, nothing)
cv2.createTrackbar('S_high2', 'Trackbars', red_upper_2[1], 255, nothing)
cv2.createTrackbar('V_high2', 'Trackbars', red_upper_2[2], 255, nothing)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 获取滑动条当前的值
    h_low1 = cv2.getTrackbarPos('H_low1', 'Trackbars')
    s_low1 = cv2.getTrackbarPos('S_low1', 'Trackbars')
    v_low1 = cv2.getTrackbarPos('V_low1', 'Trackbars')
    h_high1 = cv2.getTrackbarPos('H_high1', 'Trackbars')
    s_high1 = cv2.getTrackbarPos('S_high1', 'Trackbars')
    v_high1 = cv2.getTrackbarPos('V_high1', 'Trackbars')
    h_low2 = cv2.getTrackbarPos('H_low2', 'Trackbars')
    s_low2 = cv2.getTrackbarPos('S_low2', 'Trackbars')
    v_low2 = cv2.getTrackbarPos('V_low2', 'Trackbars')
    h_high2 = cv2.getTrackbarPos('H_high2', 'Trackbars')
    s_high2 = cv2.getTrackbarPos('S_high2', 'Trackbars')
    v_high2 = cv2.getTrackbarPos('V_high2', 'Trackbars')

    # 更新红色的上下限
    red_lower_1 = np.array([h_low1, s_low1, v_low1])
    red_upper_1 = np.array([h_high1, s_high1, v_high1])
    red_lower_2 = np.array([h_low2, s_low2, v_low2])
    red_upper_2 = np.array([h_high2, s_high2, v_high2])

    # 创建两个掩码
    mask1 = cv2.inRange(hsv, red_lower_1, red_upper_1)
    mask2 = cv2.inRange(hsv, red_lower_2, red_upper_2)
    # 合并两个掩码
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # 边缘检测
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 0:
        # 找到最大的轮廓
        cnt = max(cnts, key=cv2.contourArea)
        # 计算最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        # 绘制圆
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', res)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()