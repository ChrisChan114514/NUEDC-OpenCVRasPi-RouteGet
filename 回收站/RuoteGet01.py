import cv2
import numpy as np

def find_valid_start_point(contour, frame_height, check_region):
    """查找符合底部中间区域的有效起点"""
    bottom_point = None
    points = contour.reshape(-1, 2)
    
    # 只关注底部区域（最后20%高度）
    bottom_threshold = frame_height * 0.8
    bottom_points = points[points[:, 1] > bottom_threshold]
    
    if len(bottom_points) == 0:
        return None
    
    # 在有效区域内选择最下方的点
    valid_points = [p for p in bottom_points 
                   if check_region[0] < p[0] < check_region[1]]
    
    if not valid_points:
        return None
    
    # 取y值最大的点（最靠近底部）
    valid_points = np.array(valid_points)
    max_y = np.max(valid_points[:, 1])
    candidates = valid_points[valid_points[:, 1] == max_y]
    
    # 根据区域位置选择最左或最右的点
    if check_region[0] < width//2:
        return tuple(candidates[np.argmin(candidates[:, 0])].astype(int))
    else:
        return tuple(candidates[np.argmax(candidates[:, 0])].astype(int))

def main():
    # ...（保持之前的摄像头初始化代码）
    
    while True:
        # ...（保持之前的预处理流程）
        
        # 有效区域参数
        frame_center = width // 2
        valid_region_width = width // 4  # 中间1/4区域为有效起点区
        
        # 左右有效区域定义
        left_region = (frame_center - valid_region_width, frame_center - 20)
        right_region = (frame_center + 20, frame_center + valid_region_width)
        
        # 筛选有效轮廓
        left_contour = None
        right_contour = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 基础过滤条件
            if w < 30 or h < 30:
                continue
                
            # 检查轮廓底部是否在有效区域
            left_start = find_valid_start_point(contour, height, left_region)
            right_start = find_valid_start_point(contour, height, right_region)
            
            # 左轮廓需在左区有起点且右区无起点
            if left_start and not right_start:
                if not left_contour or cv2.contourArea(contour) > cv2.contourArea(left_contour):
                    left_contour = contour
                    
            # 右轮廓需在右区有起点且左区无起点        
            elif right_start and not left_start:
                if not right_contour or cv2.contourArea(contour) > cv2.contourArea(right_contour):
                    right_contour = contour
        
        # 绘制逻辑保持不变...
        
if __name__ == '__main__':
    main()
