#V2.4 反正能跑一些（不稳定），不管了
import cv2
import numpy as np
import time
import serial  # 用于串口通信

# 初始化串口（根据实际情况修改串口号和波特率）
ser = serial.Serial("/dev/ttyAMA0", 115200)

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
        if x + w / 2 > frame.shape[1] * 0.1 and x + w / 2 < frame.shape[1] * 0.9:
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

def scan_row(frame, contour, y, previous_mid_point):
    height, width = frame.shape[:2]
    left_points = []
    right_points = []
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
        distances = [np.linalg.norm(np.array(mid_point) - np.array(previous_mid_point)) for mid_point in current_mid_points]
        closest_index = np.argmin(distances)
        closest_mid_point = current_mid_points[closest_index]
        cv2.circle(frame, closest_mid_point, 5, (0, 255, 255), -1)  # 黄色圆圈
        return closest_mid_point
    return None

def scan_column(frame, contour, x, previous_mid_point):
    """列扫描函数，用于在指定列上寻找中点"""
    height = frame.shape[0]
    top_points = []
    bottom_points = []
    y = 0
    while y < height:
        # 如果当前点是蓝色轮廓点
        if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
            # 找到当前黑色轨迹线的最上边点
            top_point = (x, y)
            # 继续向下扫描，找到最下边点
            while y < height and cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                y += 1
            bottom_point = (x, y - 1)
            # 保存当前黑色轨迹线的上下点
            top_points.append(top_point)
            bottom_points.append(bottom_point)
        else:
            y += 1

    # 计算当前列的所有中点
    current_mid_points = []
    for top_point, bottom_point in zip(top_points, bottom_points):
        mid_y = (top_point[1] + bottom_point[1]) // 2
        mid_point = (x, mid_y)
        current_mid_points.append(mid_point)

    # 如果当前列有中点，选择距离上一列中点最近的点
    if current_mid_points:
        distances = [np.linalg.norm(np.array(mid_point) - np.array(previous_mid_point)) for mid_point in current_mid_points]
        closest_index = np.argmin(distances)
        closest_mid_point = current_mid_points[closest_index]
        cv2.circle(frame, closest_mid_point, 5, (0, 255, 255), -1)  # 黄色圆圈
        return closest_mid_point
    return None

def draw_boundary_lines(frame, first_mid_point):
    """绘制巡线边界"""
    height, width = frame.shape[:2]
    if first_mid_point is not None:
        center_x = first_mid_point[0]
        left_boundary = center_x - 40
        right_boundary = center_x + 40
        # 绘制左边界
        cv2.line(frame, (left_boundary, 0), (left_boundary, height), (0, 255, 0), 2)
        # 绘制右边界
        cv2.line(frame, (right_boundary, 0), (right_boundary, height), (0, 255, 0), 2)

def check_path_and_boundary(sorted_points, frame, upper_bound, start_point):
    """判断路径与边界的关系，并输出调整信息"""
    height, width = frame.shape[:2]
    center_x = width // 2  # 图像中心 x 坐标
    threshold = 150  # 起点偏离中心的阈值

    # 如果起点偏离中心超过阈值，进行反向调节
    if start_point is not None:
        if start_point[0] > center_x + threshold:
            # 起点在右边，向左走
            return "Left"
        elif start_point[0] < center_x - threshold:
            # 起点在左边，向右走
            return "Right"

    if len(sorted_points) > 0:
        first_mid_point = sorted_points[0]
        center_x = first_mid_point[0]
        left_boundary = center_x - 40
        right_boundary = center_x + 40

        # 遍历所有点，检查是否碰到边界
        for point in sorted_points:
            x, y = point
            # 只考虑上界以下的点
            if y > upper_bound:
                if x < left_boundary:
                    # 碰到左边界，需要向右调整
                    return "Right"
                elif x > right_boundary:
                    # 碰到右边界，需要向左调整
                    return "Left"
    # 没有碰到边界，直行
    return "Straight"

def send_control_command(direction):
    """通过串口发送控制指令"""
    # 帧头：0x00 0x40
    frame_header = bytes([0x00, 0x40])
    # 帧尾：0x23 0xFF
    frame_footer = bytes([0x23, 0xFF])

    # 中间信息
    if direction == "Right":
        middle_info = bytes([0x01])
    elif direction == "Left":
        middle_info = bytes([0x02])
    else:
        middle_info = bytes([0x00])  # 直行

    # 构造完整的消息
    message = frame_header + middle_info + frame_footer

    # 发送控制指令给单片机
    ser.write(message)

def scan_contour_for_points(frame, contour, start_point, num_scans=10):
    if contour is None or start_point is None:
        # 如果路径轮廓或起点未识别，发送直行指令
        print("直行")
        send_control_command("Straight")
        return

    height, width = frame.shape[:2]
    scan_region_top = 0  # 画幅上方75%的起始位置
    scan_region_bottom = int(height * 0.75)  # 画幅上方75%的结束位置

    # 初始化
    previous_mid_point = start_point
    mid_points = [start_point]
    scan_direction = 'row'  # 初始扫描方向为行扫描

    for _ in range(num_scans):
        if scan_direction == 'row':
            # 行扫描
            y = scan_region_bottom - (scan_region_bottom - scan_region_top) * len(mid_points) // num_scans
            closest_mid_point = scan_row(frame, contour, y, previous_mid_point)
            if closest_mid_point is None:
                # 切换到列扫描
                scan_direction = 'column'
                continue
            else:
                # 更新中点
                mid_points.append(closest_mid_point)
                previous_mid_point = closest_mid_point
        else:
            # 列扫描
            column_step = width // 19  # 平均扫描19列
            left_mid_points = []
            right_mid_points = []
            for i in range(19):
                x = i * column_step
                closest_mid_point = scan_column(frame, contour, x, previous_mid_point)
                if closest_mid_point is not None:
                    # 根据行扫描最后一个点的列坐标，将中点分为左边和右边
                    if x < previous_mid_point[0]:
                        left_mid_points.append(closest_mid_point)
                    else:
                        right_mid_points.append(closest_mid_point)

            # 判断左边和右边的中点数量
            if len(left_mid_points) > len(right_mid_points):
                # 左边中点较多，保留左边的点
                mid_points.extend(left_mid_points)
            elif len(right_mid_points) > len(left_mid_points):
                # 右边中点较多，保留右边的点
                mid_points.extend(right_mid_points)
            else:
                # 左右中点数量相同，默认路径向右
                mid_points.extend(right_mid_points)

            # 切换回行扫描
            scan_direction = 'row'
            continue

    # 从第一个中点开始，寻找最相近的点进行连接
    sorted_points = mid_points[1:]  # 忽略起点
    if len(sorted_points) > 0:
        # 从第一个中点开始连接
        for i in range(1, len(sorted_points)):
            cv2.line(frame, sorted_points[i - 1], sorted_points[i], (0, 0, 255), 2)  # 红色线条

    # 绘制巡线边界
    draw_boundary_lines(frame, sorted_points[0] if len(sorted_points) > 0 else None)

    # 计算上界
    upper_bound = sorted_points[0][1] - 100 if len(sorted_points) > 0 else 0

    # 判断路径与边界的关系
    adjustment = check_path_and_boundary(sorted_points, frame, upper_bound, start_point)
    if adjustment == "Right":
        print("右转")
        send_control_command("Right")
    elif adjustment == "Left":
        print("左转")
        send_control_command("Left")
    else:
        print("直行")
        send_control_command("Straight")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 初始化帧率计算
    prev_time = 0
    curr_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        # 计算帧率
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # 在帧上显示帧率
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
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
    ser.close()  # 关闭串口

if __name__ == '__main__':
    main()
