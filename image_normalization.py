import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

hist, bins = np.histogram(img.flatten(), 256, [0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()  

cdf_m = np.ma.masked_equal(cdf, 0)  
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')

img_eq = cdf_final[img]

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')

plt.subplot(2,2,2)
plt.title("Histogram of Original")
plt.hist(img.flatten(), 256, [0,256], color='gray')

plt.subplot(2,2,3)
plt.title("Equalized Image")
plt.imshow(img_eq, cmap='gray')

plt.subplot(2,2,4)
plt.title("Histogram After Equalization")
plt.hist(img_eq.flatten(), 256, [0,256], color='gray')

plt.tight_layout()
plt.show()
