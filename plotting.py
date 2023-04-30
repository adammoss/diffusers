# Calculate 1D power spectrum. From https://github.com/nmudur/diffusion-models-astrophysical-fields-mlps

import numpy as np


def calc_1dps_img2d(img, smoothed=0.5):
    img = np.squeeze(img)
    Nx = img.shape[0]
    kvals = np.arange(0, Nx / 2)
    fft_zerocenter = np.fft.fftshift(np.fft.fft2(img) / Nx ** 2)
    impf = abs(fft_zerocenter) ** 2.0
    x, y = np.meshgrid(np.arange(Nx), np.arange(Nx))
    R = np.sqrt((x - (Nx / 2)) ** 2 + (y - (Nx / 2)) ** 2)
    filt = lambda r: impf[(R >= r - smoothed) & (R < r + smoothed)].mean()
    mean = np.vectorize(filt)(kvals)
    return kvals, mean
