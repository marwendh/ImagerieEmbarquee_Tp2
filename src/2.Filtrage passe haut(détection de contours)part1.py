import matplotlib
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim

matplotlib.use('TkAgg')
import cv2
import numpy as np


kernel_prewitt_horizontal = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

kernel_prewitt_vertical = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

kernelCon = (1 / 25) * np.array([
    [1, 0, 0, 1, 0, 0],
    [0, 2, 2, 2, 0, 0],
    [1, 2, 5, 2, 1, 0],
    [0, 2, 2, 2, 0, 0],
    [0, 0, 1, 0, 0, 0]
])

def apply_prewitt(img):
    grad_x = cv2.filter2D(img, -1, kernel_prewitt_horizontal)
    grad_y = cv2.filter2D(img, -1, kernel_prewitt_vertical)

    grad_x = grad_x.astype(np.float32)
    grad_y = grad_y.astype(np.float32)
    magnitude = cv2.magnitude(grad_x, grad_y)

    return magnitude, grad_x, grad_y

path = r'..\circuit.tif'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)


pewittNonFiltre, xpewittNonFiltre, yNonFiltred = apply_prewitt(image)

#noisy_gaussian = (random_noise(image, mode='gaussian', var=0.01) * 255).astype(np.uint8)
filteredCon = cv2.filter2D(image, -1, kernelCon)
prewitt_filtered, xFiltred, yFiltred = apply_prewitt(filteredCon)

plt.close('all')
plt.figure(figsize=(15, 10))

print(image.shape)
images = [
    (image, "Image Originale"),
    (pewittNonFiltre, "Prewitt Non Filtrée"),
    (xpewittNonFiltre, "Composant x (Non Filtrée)"),
    (yNonFiltred, "Composant y (Non Filtrée)"),
    (image, "Image Originale"),
    (prewitt_filtered, "Prewitt Filtrée"),
    (xFiltred, "Composant x (Filtrée)"),
    (yFiltred, "Composant y (Filtrée)")
]


for i, (img, title) in enumerate(images, 1):
    plt.subplot(2, 4, i)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.show()
