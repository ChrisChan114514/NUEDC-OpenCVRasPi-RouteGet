import numpy as np
import matplotlib.pyplot as plt

# 定义波形生成函数
def generate_sine_wave(t):
    return np.sin(2 * np.pi * t)

def generate_square_wave(t):
    return np.sign(np.sin(2 * np.pi * t))

def generate_triangle_wave(t):
    return 2 * np.abs(2 * (t - np.floor(t + 0.5))) - 1

def generate_sa_wave(t):
    return np.sinc(2 * np.pi * t)

# 定义阈值法判断函数
def threshold_based_classification(wave, threshold):
    """
    使用阈值法判断波形类型
    :param wave: 输入的波形数据
    :param threshold: 阈值
    :return: 高于阈值的比例
    """
    above_threshold = np.sum(wave > threshold)  # 统计高于阈值的点数
    proportion = above_threshold / len(wave)   # 计算比例
    return proportion

# 参数设置
t = np.linspace(0, 1, 1000, endpoint=False)  # 时间轴，1秒内1000个点
threshold = 0.3  # 阈值

# 生成波形
sine_wave = generate_sine_wave(t)
square_wave = generate_square_wave(t)
triangle_wave = generate_triangle_wave(t)
sa_wave = generate_sa_wave(t)

# 计算高于阈值的比例
sine_proportion = threshold_based_classification(sine_wave, threshold)
square_proportion = threshold_based_classification(square_wave, threshold)
triangle_proportion = threshold_based_classification(triangle_wave, threshold)
sa_proportion = threshold_based_classification(sa_wave, threshold)

# 输出结果
print(f"正弦波高于阈值的比例: {sine_proportion:.2f}")
print(f"方波高于阈值的比例: {square_proportion:.2f}")
print(f"三角波高于阈值的比例: {triangle_proportion:.2f}")
print(f"Sa波高于阈值的比例: {sa_proportion:.2f}")

# 绘制波形
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, sine_wave)
plt.title("Sine Wave")
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, square_wave)
plt.title("Square Wave")
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, triangle_wave)
plt.title("Triangle Wave")
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, sa_wave)
plt.title("Sa Wave")
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
plt.legend()

plt.tight_layout()
plt.show()
