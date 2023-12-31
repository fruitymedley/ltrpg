# %%

import numpy as np
from numpy import random
import scipy.fftpack as sfft
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import sys

sys.setrecursionlimit(1000000)

# %%

x = np.linspace(0, 1, 128).reshape((-1, 1))
earth1 = (1 - x) * np.array([[20.0 / 256, 13.0 / 256, 0.0 / 256, 1]]) + x * np.array(
    [[70.0 / 256, 235.0 / 256, 52.0 / 256, 1]]
)
earth2 = (1 - x) * np.array([[70.0 / 256, 235.0 / 256, 52.0 / 256, 1]]) + x * np.array(
    [[235.0 / 256, 52.0 / 256, 52.0 / 256, 1]]
)
water1 = (1 - x) * np.array([[30.0 / 256, 33.0 / 256, 117.0 / 256, 1]]) + x * np.array(
    [[52.0 / 256, 205.0 / 256, 235.0 / 256, 1]]
)
earth = colors.LinearSegmentedColormap.from_list("earth", np.vstack((earth1, earth2)))
water = colors.LinearSegmentedColormap.from_list("water", np.vstack((water1)))

# %%

size = 1000
i1 = np.linspace(-size / 2, size / 2, size, False)
ix, iy = np.meshgrid(i1, i1)
w1 = np.linspace(-np.pi, np.pi, size, False)
wx, wy = np.meshgrid(w1, w1)

# %%

# sines = np.repeat(np.sin(np.linspace(0,2*np.pi,size,False)).reshape(1,-1), size, 0)
# cosines = np.repeat(0.5 + 0.5 * np.cos(np.linspace(0,2*np.pi,size,False)).reshape(-1,1), size, 1)
# slant = np.repeat(np.linspace(-1,1,size).reshape(-1,1), size, 1)
# image = (1 - cosines) * sines + cosines * slant
# harm = sfft.fftshift(np.fft.fft2(image))

# plt.pcolor(20*np.log(np.abs(harm)))

# %%

random.seed(4097)
four = np.sqrt(random.uniform(0, 1, (size, size))) * np.exp(
    1.0j * np.random.uniform(0, 2 * np.pi, (size, size))
)
four[int(size / 2), int(size / 2)] = 0

# %%


def filt(x, y):
    r = np.sqrt(np.square(x) + np.square(y))
    g = 1
    l0 = 1e-5  # high pass
    l1 = 4e-4  # low pass
    l2 = 4e-3  # low pass
    l3 = 1e-2  # high pass
    l4 = 1e-1  # high bump
    l5 = 4e-1  # low pass
    return (
        g
        * (1.0j * r / l0)
        / (1 + 1.0j * r / l0)
        / (1 + r * 1.0j / l1)
        / (1 + r * 1.0j / l2)
        * (1.0j * r / l3)
        / (1 + r * 1.0j / l3)
        * (1 + 1.0j * r / l4)
        / (1 + r * 1.0j / l5)
    )


h = np.vectorize(filt)(wx, wy)

# %%

elevation = np.fft.ifft2(sfft.ifftshift(four * h))
elevation /= np.max(np.abs(elevation))  # * 2

# def center(x, y):
#     r = np.sqrt(np.square(x) + np.square(y))
#     q = 0.3 * size
#     return np.square(q / (q + r))

# elevation += np.vectorize(center)(ix, iy) - 0.5

# weight = np.repeat(1 - np.power(np.linspace(-1, 1, size), 4).reshape((-1, 1)), size, 1)

# idxs = (np.linspace(0, size, size, False).astype(int), np.linspace(0,size/2,size).astype(int))
# elevation = weight * elevation + (1 - weight) * np.repeat(elevation[idxs].reshape(-1, 1), size, 1)

ws = np.fft.ifftshift(np.linspace(-np.pi, np.pi, 1000, False))
lpfs = [
    np.fft.ifft(1 / np.power(1 + 1.0j * ws / np.tan(k), 4))
    for k in np.linspace(np.pi - 0.5 * np.pi / 1000, 0.5 * np.pi / 1000, 1000, True)
]
elevation = np.array(
    [
        np.convolve(np.concatenate([e, e[:-1]]), lpf, "valid")
        for e, lpf in zip(elevation, lpfs)
    ]
)

elevation *= 6.5

# %%

ocean = np.full(elevation.shape, -1)


def flood(i, j, o):
    ocean[j, i] = o

    # North
    if j + 1 < size and elevation[j + 1, i].real < 0 and ocean[j + 1, i] == -1:
        flood(i, j + 1, o)

    # South
    if j > 0 and elevation[j - 1, i].real < 0 and ocean[j - 1, i] == -1:
        flood(i, j - 1, o)

    # East
    if elevation[j, (i + 1) % size].real < 0 and ocean[j, (i + 1) % size] == -1:
        flood((i + 1) % size, j, o)

    # West
    if (
        elevation[j, (size + i - 1) % size].real < 0
        and ocean[j, (size + i - 1) % size] == -1
    ):
        flood((size + i - 1) % size, j, o)


for i in range(size):
    for j in range(size):
        if elevation[j, i].real < -1 and ocean[j, i] == -1:
            flood(i, j, np.max(ocean) + 1)

ocean = (ocean >= 0).astype(int)

# %%

plt.pcolormesh(
    elevation.real,
    cmap=earth,
)
cbar = plt.colorbar()
plt.clim(-6.5, 6.5)

plt.pcolormesh(np.where(ocean, elevation.real, np.nan), cmap=water)
cbar = plt.colorbar()
plt.clim(-6.5, 0)

plt.contour(
    elevation.real, levels=np.linspace(-6.5, 6.5, 11), colors="black", linewidths=0.1
)
# cbar.set_ticks([-1, 0, 1])

# %%

np.save("world/elevation.npy", elevation.real.transpose())
np.save("world/ocean.npy", ocean.transpose())
