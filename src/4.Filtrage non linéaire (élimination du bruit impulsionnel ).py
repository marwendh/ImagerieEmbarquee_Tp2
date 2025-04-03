import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


img= cv2.imread(r'..\cellSP.tif', cv2.IMREAD_GRAYSCALE)
img_median =  cv2.medianBlur(img,3)
img_moyen = cv2.blur(img,(3,3))

images = [
    (img, "originale"),
    (img_median, " filtre median"),
    (img_moyen, " filtre moyen"),
]

plt.figure(figsize=(15, 10))

for i, (img, title) in enumerate(images, 1):
    plt.subplot(1, 3, i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
