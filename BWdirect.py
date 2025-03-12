import cv2
import numpy as np

def find_path_contour(frame, dilated):
    # 寻找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 初始化路径和最小距离
    path = None
    min_distance = float('inf')
    
    # 筛选出位于图像中央的轮廓作为路径
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # 确保轮廓位于图像中央
        if x + w / 2 > frame.shape[1] * 0.25 and x + w / 2 < frame.shape[1] * 0.75:
            # 计算轮廓中心到图像底部的距离
            distance = frame.shape[0] - y - h
            if distance < min_distance:
                min_distance = distance
                path = contour
    return path

def draw_contour_path(frame, contour, color):
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, color, 2)

def find_start_point(frame, contour):
    if contour is not None:
        # 找到轮廓的最底部点
        bottom_point = None
        for point in contour:
            x, y = point[0]
            if bottom_point is None or y > bottom_point[1]:
                bottom_point = (x, y)
        return bottom_point
    return None

def scan_contour_for_points(frame, contour, start_point, num_scans=10):
    if contour is not None:
        height, width = frame.shape[:2]
        scan_region_top = 0  # 画幅上方75%的起始位置
        scan_region_bottom = int(height * 0.75)  # 画幅上方75%的结束位置

        # 从下往上逐行扫描
        previous_mid_point = start_point  # 初始化为起点
        mid_points = []  # 保存所有筛选后的中点
        for i in range(num_scans):
            y = scan_region_bottom - (scan_region_bottom - scan_region_top) * i // (num_scans - 1)
            # 初始化当前行的所有黑色轨迹线的左右点
            left_points = []
            right_points = []
            # 从左至右扫描
            x = 0
            while x < width:
                # 如果当前点是蓝色轮廓点
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    # 找到当前黑色轨迹线的最左边点
                    left_point = (x, y)
                    # 继续向右扫描，找到最右边点
                    while x < width and cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                        x += 1
                    right_point = (x - 1, y)
                    # 保存当前黑色轨迹线的左右点
                    left_points.append(left_point)
                    right_points.append(right_point)
                else:
                    x += 1

            # 计算当前行的所有中点
            current_mid_points = []
            for left_point, right_point in zip(left_points, right_points):
                mid_x = (left_point[0] + right_point[0]) // 2
                mid_point = (mid_x, y)
                current_mid_points.append(mid_point)

            # 如果当前行有中点，选择距离上一行中点最近的点
            if current_mid_points:
                # 计算每个中点与上一行中点的距离
                distances = [np.linalg.norm(np.array(mid_point) - np.array(previous_mid_point)) for mid_point in current_mid_points]
                # 找到距离最近的点
                closest_index = np.argmin(distances)
                closest_mid_point = current_mid_points[closest_index]
                # 标记最近的中点
                cv2.circle(frame, closest_mid_point, 5, (0, 255, 255), -1)  # 黄色圆圈
                # 保存当前行的中点
                mid_points.append(closest_mid_point)
                # 更新上一行中点
                previous_mid_point = closest_mid_point

        # 连接所有中点
        if len(mid_points) > 1:
            for i in range(1, len(mid_points)):
                cv2.line(frame, mid_points[i - 1], mid_points[i], (0, 255, 0), 2)  # 绿色线条连接中点

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        height, width = frame.shape[:2]
        cropped_binary = binary[:int(height * 0.75), :]
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(cropped_binary, kernel, iterations=1)

        # 找到路径轮廓
        path_contour = find_path_contour(frame, dilated)
        draw_contour_path(frame, path_contour, (255, 0, 0))  # 蓝色轮廓

        # 找到起点
        start_point = find_start_point(frame, path_contour)
        if start_point is not None:
            # 绘制绿色十字标记起点
            cv2.drawMarker(frame, start_point, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # 在画幅上方75%的区域扫描蓝色轮廓，标记每一行内所有黑色轨迹线的左右点和中点
        scan_contour_for_points(frame, path_contour, start_point, num_scans=10)

        cv2.imshow('Frame', frame)
        cv2.imshow('Edges', dilated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
