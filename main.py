import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

# Đọc ảnh
img1 = cv2.imread(r'C:\Users\Admin\PycharmProjects\task8\input\anh1.jpg', 0)
img2 = cv2.imread(r'C:\Users\Admin\PycharmProjects\task8\input\anh2.jpg', 0)

def apply_filters(img, title_prefix):
    # 1. Gaussian filter (Blurring)
    gaussian_img = cv2.GaussianBlur(img, (5, 5), 0)

    # 2. Sobel filter
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)

    # 3. Prewitt filter
    prewittx = ndimage.prewitt(img, axis=0)
    prewitty = ndimage.prewitt(img, axis=1)
    prewitt_combined = np.hypot(prewittx, prewitty)

    # 4. Roberts filter
    robertsx = np.array([[1, 0], [0, -1]])
    robertsy = np.array([[0, 1], [-1, 0]])
    robertsx_img = ndimage.convolve(img, robertsx)
    robertsy_img = ndimage.convolve(img, robertsy)
    roberts_combined = np.hypot(robertsx_img, robertsy_img)

    # 5. Canny edge detection
    canny_img = cv2.Canny(img, 100, 200)

    # Hiển thị các ảnh kết quả
    images = [img, gaussian_img, sobel_combined, prewitt_combined, roberts_combined, canny_img]
    titles = ['Original Image', 'Gaussian Blurred', 'Sobel', 'Prewitt', 'Roberts', 'Canny']

    plt.figure(figsize=(10, 8))
    for i in range(len(images)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"{title_prefix} - {titles[i]}")
        plt.xticks([]), plt.yticks([])
    plt.show()

# Áp dụng cho từng ảnh
apply_filters(img1, "Image 1")
apply_filters(img2, "Image 2")
