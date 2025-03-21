import cv2
import numpy as np

# 读取图像
image_path = "C:\Users\ChrisChan\Desktop\1.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 计算白色区域的轮廓
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 计算线条宽度
widths = []
for contour in contours:
    for i in range(len(contour) - 1):
        p1, p2 = contour[i][0], contour[i + 1][0]
        dist = np.linalg.norm(p1 - p2)
        widths.append(dist)

# 计算线宽范围
if widths:
    min_width = min(widths)
    max_width = max(widths)
else:
    min_width, max_width = 0, 0

min_width, max_width
