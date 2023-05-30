# Calculate 1D power spectrum and Minkowski functionals.
# From https://github.com/nmudur/diffusion-models-astrophysical-fields-mlps

import numpy as np
from quantimpy import minkowski as mk


def calc_1dps_img2d(img1, img2=None, smoothed=0.5):
    img1 = np.squeeze(img1)
    Nx = img1.shape[0]
    kvals = np.arange(0, Nx / 2)
    if img2 is None:
        fft_zerocenter = np.fft.fftshift(np.fft.fft2(img1) / Nx ** 2)
        impf = np.real(fft_zerocenter) ** 2 + np.imag(fft_zerocenter) ** 2
    else:
        img2 = np.squeeze(img2)
        assert img1.shape == img2.shape
        fft_zerocenter1 = np.fft.fftshift(np.fft.fft2(img1) / Nx ** 2)
        fft_zerocenter2 = np.fft.fftshift(np.fft.fft2(img2) / Nx ** 2)
        impf = np.real(fft_zerocenter1) * np.real(fft_zerocenter2) + \
               np.imag(fft_zerocenter1) * np.imag(fft_zerocenter2)
    x, y = np.meshgrid(np.arange(Nx), np.arange(Nx))
    R = np.sqrt((x - (Nx / 2)) ** 2 + (y - (Nx / 2)) ** 2)
    filt = lambda r: impf[(R >= r - smoothed) & (R < r + smoothed)].mean()
    mean = np.vectorize(filt)(kvals)
    return kvals, mean


def get_minkowski(img, min=-1, max=1):
    img = np.squeeze(img)
    gs_vals = np.linspace(min, max, 50)
    gs_masks = [img >= gs_vals[ig] for ig in range(len(gs_vals))]
    minkowski = []
    for i in range(len(gs_masks)):
        minkowski.append(mk.functionals(gs_masks[i], norm=True))
    return np.vstack(minkowski)
