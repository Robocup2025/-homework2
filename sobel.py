import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. 自定义二维卷积函数
def convolve2d(image, kernel):
    """手动实现二维卷积，不使用 OpenCV 内置函数"""
    # 翻转卷积核
    kernel = np.flipud(np.fliplr(kernel))
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # 对图像做边缘填充
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    # 输出初始化
    output = np.zeros_like(image, dtype=np.float32)

    # 卷积计算
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

# 2. 定义 Sobel 核
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

# 3. 读取图像并灰度化
img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("❌ 未找到图像，请将图片命名为 lena.png 并与脚本放在同目录。")

# 4. 卷积计算\
Gx = convolve2d(img, sobel_x)
Gy = convolve2d(img, sobel_y)

# 5. 计算梯度幅值与方向
magnitude = np.sqrt(Gx**2 + Gy**2)
direction = np.arctan2(Gy, Gx)

# 归一化到 0-255
def normalize(img):
    img = img - img.min()
    img = img / img.max() * 255
    return img.astype(np.uint8)

Gx_n = normalize(Gx)
Gy_n = normalize(Gy)
mag_n = normalize(magnitude)
dir_n = normalize(direction)

# 6. 可视化结果
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Sobel X")
plt.imshow(Gx_n, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Sobel Y")
plt.imshow(Gy_n, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Gradient Magnitude")
plt.imshow(mag_n, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Gradient Direction (arctan2)")
plt.imshow(dir_n, cmap='hsv')
plt.axis('off')

plt.tight_layout()
plt.show()
