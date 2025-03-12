import cv2
import numpy as np

class PathTracker:
    def __init__(self):
        # 初始化参数
        self.cap = cv2.VideoCapture(0)
        self.frame_width = 640
        self.frame_height = 480
        self.car_position = (self.frame_width//2, self.frame_height-20)  # 小车初始位置
        self.path_color = (0, 0, 255)  # 红色路径
        self.debug_mode = False
        
        # 参数配置
        self.params = {
            'black_threshold': 50,       # 黑色阈值
            'min_path_width': 20,        # 最小路径宽度
            'max_path_width': 300,       # 最大路径宽度
            'car_region_height': 0.2,    # 车头区域高度比例
            'look_ahead_steps': 5,      # 路径采样点数
            'min_path_length': 50,       # 最小路径长度
            'steering_gain': 1.5,        # 转向增益
            'morph_kernel_size': 7,      # 形态学核大小
            'dilate_iterations': 2       # 膨胀迭代次数
        }
        
        # 检查摄像头
        if not self.cap.isOpened():
            print("无法打开摄像头")
            exit()

    def preprocess_frame(self, frame):
        """图像预处理"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 形态学操作
        kernel = np.ones((self.params['morph_kernel_size'], 
                        self.params['morph_kernel_size']), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, 
                                    iterations=self.params['dilate_iterations'])
        
        return processed

    def find_start_point(self, contour):
        """查找路径起点"""
        # 获取车头区域
        car_region_top = int(self.frame_height * (1 - self.params['car_region_height']))
        car_region_bottom = self.frame_height
        
        # 获取轮廓点
        points = contour.reshape(-1, 2)
        
        # 筛选车头区域内的点
        in_region_points = [p for p in points if car_region_top < p[1] <= car_region_bottom]
        
        if not in_region_points:
            return None
        
        # 找到最靠近底部的点
        in_region_points = np.array(in_region_points)
        max_y = np.max(in_region_points[:, 1])
        candidates = in_region_points[in_region_points[:, 1] == max_y]
        
        # 取中间点作为起点
        return tuple(candidates[len(candidates)//2].astype(int))

    def validate_path(self, contour):
        """验证路径有效性"""
        # 检查路径长度
        if cv2.arcLength(contour, False) < self.params['min_path_length']:
            return False
            
        # 检查路径宽度
        x, y, w, h = cv2.boundingRect(contour)
        if not (self.params['min_path_width'] < w < self.params['max_path_width']):
            return False
            
        # 检查起点是否在车头区域
        start_point = self.find_start_point(contour)
        if start_point is None:
            return False
            
        return True

    def sample_path_points(self, contour):
        """沿路径采样关键点"""
        # 获取轮廓点
        points = contour.reshape(-1, 2)
        
        # 按y坐标排序
        sorted_points = sorted(points, key=lambda p: p[1])
        
        # 等间隔采样
        step = max(1, len(sorted_points) // self.params['look_ahead_steps'])
        sampled_points = [sorted_points[i*step] for i in range(self.params['look_ahead_steps'])]
        
        return np.array(sampled_points,dtype=np.int32)


    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取帧")
                break
            
            # 图像预处理
            processed = self.preprocess_frame(frame)
            
            # 查找所有轮廓
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 筛选有效路径
            valid_path = None
            for contour in contours:
                if self.validate_path(contour):
                    valid_path = contour
                    break
            
            # 路径规划
            steering = 0
            path_points = []
            if valid_path is not None:
                path_points = self.sample_path_points(valid_path)
                steering = self.calculate_steering(path_points)
            
            # 可视化
            output_frame = self.visualize(frame, path_points, steering)
            
            # 显示结果
            cv2.imshow('Path Tracking', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    tracker = PathTracker()
    tracker.run()
