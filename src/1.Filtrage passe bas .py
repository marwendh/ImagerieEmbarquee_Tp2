import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim

kernelPyr = (1/81)*np.array([
    [1, 2, 3, 2, 1],
    [2, 4, 6, 4, 2],
    [3, 6, 9, 6, 3],
    [2, 4, 6, 4, 2],
    [1, 2, 3, 2, 1]
])

kernelBin = (1/256)*np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
])

kernelCon = (1/25)*np.array([
    [1, 0, 0, 1, 0, 0],
    [0, 2, 2, 2, 0, 0],
    [1, 2, 5, 2, 1, 0],
    [0, 2, 2, 2, 0, 0],
    [0, 0, 1, 0, 0, 0]
])

def compute_ssim(original, filtered):
    return ssim(original, filtered, data_range=255)

def compute_psnr(original, filtered):
    mse = np.mean((cv2.subtract(original, filtered)) ** 2)
    if mse == 0:
        return float('inf')  # Images are identical
    return 10 * np.log10((255**2) / mse)

def compute_snr(original, noisy):
    """Calcule le rapport signal sur bruit (SNR) entre l'image originale et l'image bruitée/filtrée."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((cv2.subtract(original, noisy)) ** 2)

    if noise_power == 0:
        return float('inf')  # Avoid division by zero

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def apply_filters(noisy_img):
    # Apply various filters
    filteredPyr = cv2.filter2D(noisy_img, -1, kernelPyr)
    filteredBin = cv2.filter2D(noisy_img, -1, kernelBin)
    filteredCon = cv2.filter2D(noisy_img, -1, kernelCon)

    gaussian_filtered = cv2.GaussianBlur(noisy_img, (5, 5), 0)  # Apply Gaussian filter
    bilateral_filtered = cv2.bilateralFilter(noisy_img, 9, 75, 75)  # Apply Bilateral filter

    return filteredPyr, filteredBin, filteredCon, gaussian_filtered, bilateral_filtered

path = r'..\circuit.tif'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Add noise to the image
noisy_gaussian = (random_noise(image, mode='gaussian', var=0.01) * 255).astype(np.uint8)
noisy_poisson = (random_noise(image, mode='poisson') * 255).astype(np.uint8)
noisy_speckle = (random_noise(image, mode='speckle', var=0.01) * 255).astype(np.uint8)

# Select a noisy image
selected_noisy = noisy_poisson

# Apply filters
mean_filtered, gaussian_filtered, bilateral_filtered, pyr_filtered, bin_filtered = apply_filters(selected_noisy)

# Display the images
cv2.imshow('Image originale', image)
cv2.imshow('Image bruitée', selected_noisy)
cv2.imshow('Filtrage Pyr', pyr_filtered)
cv2.imshow('Filtrage Bin', bin_filtered)
cv2.imshow('Filtrage Gaussien', gaussian_filtered)
cv2.imshow('Filtrage Bilateral', bilateral_filtered)

# Compute and print metrics
print(f"SNR avant filtrage : {compute_snr(image, selected_noisy):.2f} dB")
print(f"SNR après filtrage (Pyramidal) : {compute_snr(image, pyr_filtered):.2f} dB")
print(f"SNR après filtrage (Binomial) : {compute_snr(image, bin_filtered):.2f} dB")
print(f"SNR après filtrage (Gaussien) : {compute_snr(image, gaussian_filtered):.2f} dB")
print(f"SNR après filtrage (Bilatéral) : {compute_snr(image, bilateral_filtered):.2f} dB")

print(f"PSNR avant filtrage : {compute_psnr(image, selected_noisy):.2f} dB")
print(f"PSNR (Pyramidal) : {compute_psnr(image, pyr_filtered):.2f} dB")
print(f"PSNR (Binomial) : {compute_psnr(image, bin_filtered):.2f} dB")
print(f"PSNR (Gaussien) : {compute_psnr(image, gaussian_filtered):.2f} dB")
print(f"PSNR (Bilatéral) : {compute_psnr(image, bilateral_filtered):.2f} dB")

print(f"SSIM avant filtrage : {compute_ssim(image, selected_noisy):.2f}")
print(f"SSIM (Pyramidal) : {compute_ssim(image, pyr_filtered):.2f}")
print(f"SSIM (Binomial) : {compute_ssim(image, bin_filtered):.2f}")
print(f"SSIM (Gaussien) : {compute_ssim(image, gaussian_filtered):.2f}")
print(f"SSIM (Bilatéral) : {compute_ssim(image, bilateral_filtered):.2f}")

cv2.waitKey(0)
cv2.destroyAllWindows()
