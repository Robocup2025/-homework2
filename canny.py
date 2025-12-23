import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1️ 高斯滤波（降噪）
def gaussian_kernel(size=5, sigma=1.0):
    """生成高斯卷积核"""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def convolve2d(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))  
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    output = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output


# 2️ 计算梯度幅值与方向（Sobel）
def compute_gradient(img):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]], dtype=np.float32)
    Ix = convolve2d(img, sobel_x)
    Iy = convolve2d(img, sobel_y)
    magnitude = np.sqrt(Ix**2 + Iy**2)
    direction = np.arctan2(Iy, Ix)
    return Ix, Iy, magnitude, direction


# 3️.非极大值抑制（NMS）
def non_max_suppression(magnitude, direction):
    """沿梯度方向抑制非极大值"""
    M, N = magnitude.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0
    return Z


# 4 双阈值检测与边缘连接
def double_threshold_and_hysteresis(img, low_ratio=0.05, high_ratio=0.15):
    high = img.max() * high_ratio
    low = high * low_ratio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.uint8)

    strong = 255
    weak = 75

    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # 边缘连接：弱边缘如果邻近强边缘则保留
    for i in range(1, M-1):
        for j in range(1, N-1):
            if res[i, j] == weak:
                if np.any(res[i-1:i+2, j-1:j+2] == strong):
                    res[i, j] = strong
                else:
                    res[i, j] = 0
    return res


# 主程序：Canny完整流程
def manual_canny(image_path, sigma=1.4):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("未找到图像文件")

 
    kernel = gaussian_kernel(size=5, sigma=sigma)
    smoothed = convolve2d(img, kernel)

    Ix, Iy, mag, direction = compute_gradient(smoothed)

    nms = non_max_suppression(mag, direction)

    edge = double_threshold_and_hysteresis(nms)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1); plt.title("Original"); plt.imshow(img, cmap='gray'); plt.axis('off')
    plt.subplot(2, 3, 2); plt.title("Gaussian Smoothed"); plt.imshow(smoothed, cmap='gray'); plt.axis('off')
    plt.subplot(2, 3, 3); plt.title("Gradient Magnitude"); plt.imshow(mag, cmap='gray'); plt.axis('off')
    plt.subplot(2, 3, 4); plt.title("NMS"); plt.imshow(nms, cmap='gray'); plt.axis('off')
    plt.subplot(2, 3, 5); plt.title("Final Edge"); plt.imshow(edge, cmap='gray'); plt.axis('off')
    plt.tight_layout()
    plt.show()

    return edge

# 运行
if __name__ == "__main__":
    manual_canny("lena.png", sigma=1.4)
