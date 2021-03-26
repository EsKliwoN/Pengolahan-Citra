import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp

def distance(point1, point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterLP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < D0:
                base[y, x] = 1
    return base

def idealFilterHP(D0, imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < D0:
                base[y, x] = 0
    return base

def butterworthLP(D0, imgShape, n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1/(1+(distance((y, x), center)/D0)**(2*n))
    return base

def butterworthHP(D0, imgShape, n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1-1/(1+(distance((y, x), center)/D0)**(2*n))
    return base

def gaussianLP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = exp(((-distance((y, x), center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 - exp(((-distance((y, x), center)**2)/(2*(D0**2))))
    return base

# Filter Value
LPIdealValue = 50
LPBwValue = 50
LPGaussianValue = 50
HPIdealValue = 50
HPBwValue = 50
HPGaussianValue = 50

nButterWorth = 10

# Fourier Transform
img = cv2.imread("im.jpg", 0)
img_c1 = np.fft.fft2(img)
phase = np.fft.fft2(img)
img_c2 = np.fft.fftshift(img_c1)
img_c3 = np.fft.ifftshift(img_c2)
img_c4 = np.fft.ifft2(img_c3)

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
plt.figure(1).suptitle("Fast Fourier Transform", fontsize=20)
plt.subplot(231), plt.imshow(img, "gray"), plt.title("Original Image")
plt.subplot(232), plt.imshow(np.log(1+np.abs(img_c1)), "gray"), plt.title("Spectrum")
plt.subplot(233), plt.imshow(np.angle(phase), "gray"), plt.title("Phase Angle")
plt.subplot(234), plt.imshow(np.log(1+np.abs(img_c2)), "gray"), plt.title("Centered Spectrum")
plt.subplot(235), plt.imshow(np.log(1+np.abs(img_c3)), "gray"), plt.title("Decentralized")
plt.subplot(236), plt.imshow(np.abs(img_c4), "gray"), plt.title("Processed Image")

original = np.fft.fft2(img)
center = np.fft.fftshift(original)


# Ideal
IdealLP = center * idealFilterLP(LPIdealValue, img.shape)
LowPass1 = np.fft.ifftshift(IdealLP)
inverse_LowPass1 = np.fft.ifft2(LowPass1)
FilterIdealLP = idealFilterLP(LPIdealValue, img.shape)
FilterIdealLPAF = center * FilterIdealLP

IdealHP = center * idealFilterHP(HPIdealValue, img.shape)
HighPass1 = np.fft.ifftshift(IdealHP)
inverse_HighPass1 = np.fft.ifft2(HighPass1)
FilterIdealHP = idealFilterHP(HPIdealValue, img.shape)
FilterIdealHPAF = center * FilterIdealHP

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
plt.figure(2), plt.figure(2).suptitle("Ideal", fontsize=20)
plt.subplot(2, 5, 1), plt.imshow(img, "gray"), plt.title("Original Image")
plt.subplot(2, 5, 2), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Magnitude")
plt.subplot(2, 5, 3), plt.imshow(np.abs(FilterIdealLP), "gray"), plt.title("Low Pass")
plt.subplot(2, 5, 4), plt.imshow(np.log(1+np.abs(FilterIdealLPAF)), "gray"), plt.title("Magnitude after LPF")
plt.subplot(2, 5, 5), plt.imshow(np.abs(inverse_LowPass1), "gray"), plt.title("Processed Image")
plt.subplot(2, 5, 6), plt.imshow(img, "gray"), plt.title("Original Image")
plt.subplot(2, 5, 7), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Magnitude")
plt.subplot(2, 5, 8), plt.imshow(np.abs(FilterIdealHP), "gray"), plt.title("High Pass")
plt.subplot(2, 5, 9), plt.imshow(np.log(1+np.abs(FilterIdealHPAF)), "gray"), plt.title("Magnitude after HPF")
plt.subplot(2, 5, 10), plt.imshow(np.abs(inverse_HighPass1), "gray"), plt.title("Processed Image")


# Butterworth
BwLP = center * butterworthLP(LPBwValue, img.shape, nButterWorth)
LowPass2 = np.fft.ifftshift(BwLP)
inverse_LowPass2 = np.fft.ifft2(LowPass2)
FilterButterLP = butterworthLP(LPBwValue, img.shape, nButterWorth)
FilterButterLPAF = center * FilterButterLP

BwHP = center * butterworthHP(HPBwValue, img.shape, nButterWorth)
HighPass2 = np.fft.ifftshift(BwHP)
inverse_HighPass2 = np.fft.ifft2(HighPass2)
FilterButterHP = butterworthHP(HPBwValue, img.shape, nButterWorth)
FilterButterHPAF = center * FilterButterHP

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
plt.figure(3), plt.figure(3).suptitle("Butterworth", fontsize=20)
plt.subplot(2, 5, 1), plt.imshow(img, "gray"), plt.title("Original Image")
plt.subplot(2, 5, 2), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Magnitude")
plt.subplot(2, 5, 3), plt.imshow(np.abs(FilterButterLP), "gray"), plt.title("Low Pass")
plt.subplot(2, 5, 4), plt.imshow(np.log(1+np.abs(FilterButterLPAF)), "gray"), plt.title("Magnitude after LPF")
plt.subplot(2, 5, 5), plt.imshow(np.abs(inverse_LowPass2), "gray"), plt.title("Processed Image")
plt.subplot(2, 5, 6), plt.imshow(img, "gray"), plt.title("Original Image")
plt.subplot(2, 5, 7), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Magnitude")
plt.subplot(2, 5, 8), plt.imshow(np.abs(FilterButterHP), "gray"), plt.title("High Pass")
plt.subplot(2, 5, 9), plt.imshow(np.log(1+np.abs(FilterButterHPAF)), "gray"), plt.title("Magnitude after HPF")
plt.subplot(2, 5, 10), plt.imshow(np.abs(inverse_HighPass2), "gray"), plt.title("Processed Image")


# Gaussian
GaussianLP = center * gaussianLP(LPGaussianValue, img.shape)
LowPass3 = np.fft.ifftshift(GaussianLP)
inverse_LowPass3 = np.fft.ifft2(LowPass3)
FilterGaussianLP = gaussianLP(LPGaussianValue, img.shape)
FilterGaussianLPAF = center * FilterGaussianLP

GaussianHP = center * gaussianHP(HPGaussianValue, img.shape)
HighPass3 = np.fft.ifftshift(GaussianHP)
inverse_HighPass3 = np.fft.ifft2(HighPass3)
FilterGaussianHP = gaussianHP(HPGaussianValue, img.shape)
FilterGaussianHPAF = center * FilterGaussianHP

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
plt.figure(4), plt.figure(4).suptitle("Gaussian", fontsize=20)
plt.subplot(2, 5, 1), plt.imshow(img, "gray"), plt.title("Original Image")
plt.subplot(2, 5, 2), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Magnitude")
plt.subplot(2, 5, 3), plt.imshow(np.abs(FilterGaussianLP), "gray"), plt.title("Low Pass")
plt.subplot(2, 5, 4), plt.imshow(np.log(1+np.abs(FilterGaussianLPAF)), "gray"), plt.title("Magnitude after LPF")
plt.subplot(2, 5, 5), plt.imshow(np.abs(inverse_LowPass3), "gray"), plt.title("Processed Image")
plt.subplot(2, 5, 6), plt.imshow(img, "gray"), plt.title("Original Image")
plt.subplot(2, 5, 7), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Magnitude")
plt.subplot(2, 5, 8), plt.imshow(np.abs(FilterGaussianHP), "gray"), plt.title("High Pass")
plt.subplot(2, 5, 9), plt.imshow(np.log(1+np.abs(FilterGaussianHPAF)), "gray"), plt.title("Magnitude after HPF")
plt.subplot(2, 5, 10), plt.imshow(np.abs(inverse_HighPass3), "gray"), plt.title("Processed Image")

plt.show()