import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读入图像
img = cv2.imread('lena.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# 2. Sobel 计算梯度
Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# 3. 计算二阶矩阵 M 的分量
Ix2 = Ix ** 2
Iy2 = Iy ** 2
Ixy = Ix * Iy

# 4. 用高斯滤波平滑
Ix2 = cv2.GaussianBlur(Ix2, (5,5), sigmaX=1)
Iy2 = cv2.GaussianBlur(Iy2, (5,5), sigmaX=1)
Ixy = cv2.GaussianBlur(Ixy, (5,5), sigmaX=1)

# 5. 计算 Harris 响应
k = 0.04
R = (Ix2 * Iy2 - Ixy ** 2) - k * (Ix2 + Iy2) ** 2

# 6. 归一化
R_norm = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
R_norm = np.uint8(R_norm)

# 7. 角点检测（阈值 + 非极大值）
corner_thresh = 0.01 * R.max()
corners = np.zeros_like(R_norm)
corners[R > corner_thresh] = 255

# 8. 将角点绘制在原图上
result = img.copy()
result[R > corner_thresh] = [0, 0, 255]  # 红色角点

# 9. 可视化结果
plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.title('Gray Image')
plt.imshow(gray, cmap='gray')

plt.subplot(1,3,2)
plt.title('Harris Response R')
plt.imshow(R, cmap='jet')

plt.subplot(1,3,3)
plt.title('Detected Corners')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()
