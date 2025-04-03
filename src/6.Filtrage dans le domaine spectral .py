import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

path = r'..\lenabruitee.png'
img = cv.imread(path, 0)

pathmasq = r'..\masqueLena.tif'
maskI = cv.imread(pathmasq, 0)
# apply DFT
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
rows, cols = img.shape


mask = np.zeros((rows, cols, 2), np.uint8)
mask[:, :, 0] = maskI[:, :]
mask[:, :, 1] = maskI[:, :]


fshift = dft_shift * mask

f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('filtered image'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)
# closing all open windows
cv.destroyAllWindows()