import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
I = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(I, (50, 50), (150, 150), 255, -1)

kernel_prewitt_horizontal = np.array([[-1, 0, 1],
                                      [-1, 0, 1],
                                      [-1, 0, 1]])
kernel_prewitt_vertical = np.array([[-1, -1, -1],
                                       [0, 0, 0],
                                       [1, 1, 1]])

kernel_sobel_horizontal = np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]])
kernel_sobel_vertical = np.array([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]])

grad_x_prewitt = cv2.filter2D(I, cv2.CV_64F, kernel_prewitt_horizontal)
grad_y_prewitt = cv2.filter2D(I, cv2.CV_64F, kernel_prewitt_vertical)
magnitude_prewitt = np.sqrt(grad_x_prewitt**2+ grad_y_prewitt**2)

grad_x_sobel = cv2.filter2D(I, cv2.CV_64F, kernel_sobel_horizontal)
grad_y_sobel = cv2.filter2D(I, cv2.CV_64F, kernel_sobel_vertical)
magnitude_sobel = np.sqrt(grad_x_sobel**2+ grad_y_sobel**2)

#Canny filter

I= cv2.imread(r'..\gantrycrane.png', cv2.IMREAD_GRAYSCALE)
img_canny = cv2.Canny(I, 50, 150)

images = [
    (magnitude_prewitt, "prewitt"),
    (grad_x_prewitt, " x avec Prewitt"),
    (grad_y_prewitt, " y avec Prewitt"),
     (magnitude_sobel, "sobel"),
    (grad_x_sobel, "x avec Sobel"),
    (grad_y_sobel, "y avec Sobel"),

]

plt.figure(figsize=(15, 10))

for i, (img, title) in enumerate(images, 1):
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
