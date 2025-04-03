import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

img= cv2.imread(r'..\blobs.png', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

ersosion=cv2.erode(img,kernel,iterations = 1)
dilatation=cv2.dilate(img,kernel,iterations = 3)
open=cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
close=cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

images = [
    (img, "originale"),
    ( ersosion, "ersosion"),
    (dilatation, "dilatation"),
    (open, "open"),
    (close, "close")

]

plt.figure(figsize=(15, 10))

for i, (img, title) in enumerate(images, 1):
    plt.subplot(1, 5, i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
