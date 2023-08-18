from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
from scipy import signal
from scipy import interpolate
from colorsys import hsv_to_rgb

# %%

df = pd.read_csv("assets/globe2.csv")
elevation = df.to_numpy()[:, 1:].transpose()
Xsize, Ysize = elevation.shape

# %%

longitudes = np.linspace(
    2 * np.pi / (2 * Xsize), 2 * np.pi * (1 - 1 / (2 * Xsize)), Xsize, False
)
latitudes = np.linspace(
    -np.pi / 2 * ((2 * Ysize - 1) / (2 * Ysize)),
    np.pi / 2 * ((2 * Ysize - 1) / (2 * Ysize)),
    Ysize,
    True,
)

X, Y = np.meshgrid(longitudes, latitudes, indexing="ij")

# %%

maxWind = 50.0 / 3600
cellsPerHemisphere = 3
rotation = -1

# %%

VelX = (
    maxWind
    * rotation
    * signal.square(Y)
    * signal.square(Y * cellsPerHemisphere, 0.5)
    * np.cos(Y * cellsPerHemisphere)
)
VelY = -maxWind * np.sin(2 * Y * cellsPerHemisphere)

Humidity = np.square(np.cos(Y * cellsPerHemisphere))

dhdx = (np.roll(elevation, -1, axis=0) - np.roll(elevation, 1, axis=0)) / 2
dhdy = np.concatenate(
    (np.maximum(elevation, 0)[:, 1:], np.maximum(elevation, 0)[:, -1:]), axis=1
) - np.concatenate(
    (np.maximum(elevation, 0)[:, :1], np.maximum(elevation, 0)[:, :-1]), axis=1
)


def treck(x, y):
    h = 0
    decay = 1
    # decay = amount = 0.95
    for i in range(100):
        i, j = min(Xsize - 1, max(0, int(Xsize * x / (2 * np.pi)))), min(
            Ysize - 1, max(0, int(Ysize * (y / np.pi + 0.5)))
        )
        vx = VelX[i, j]
        vy = VelY[i, j]
        dx = dhdx[i, j]
        dy = dhdy[i, j]
        decay *= min(1, np.exp(-(vx * dx + vy * dy) / maxWind))
        x -= vx / 3
        y -= vy / 3
        if x < 0:
            x = 2 * np.pi + x
        elif x >= 2 * np.pi:
            x = x - 2 * np.pi
        if y < -np.pi / 2:
            x = 2 * np.pi - x
            y = -np.pi / 2 - (-np.pi / 2 + y)
        elif y >= np.pi / 2:
            x = 2 * np.pi - x
            y = np.pi / 2 - (y - np.pi / 2)

        if elevation[i, j] < 0:
            h += 1 * np.cos(y) * decay
        # amount *= decay
    return h


humidity = np.vectorize(treck)(X, Y)
print("Done")


filterSize = 50
decay = 0.02
filter = (
    np.square(decay)
    / (2 * np.pi)
    * np.exp(
        -decay
        * np.sqrt(
            np.sum(
                np.square(
                    np.mgrid[
                        -filterSize : filterSize + 1 : 1,
                        -filterSize : filterSize + 0.5 : 0.5,
                    ]
                ),
                axis=0,
            )
        )
    )
)
wrap = np.concatenate((humidity[-filterSize:], humidity, humidity[:filterSize]), axis=0)
wrap = np.concatenate(
    (
        wrap[::-1, 2 * filterSize - 1 :: -1],
        wrap,
        wrap[::-1, : -2 * filterSize - 1 : -1],
    ),
    axis=1,
)
humidity = (elevation > 0).astype(int) * signal.convolve2d(filter, wrap, "valid")
humidity = (100 - 100.0 / 2**8) / (humidity.max() - humidity.min()) * (
    humidity - humidity.min()
) + 100.0 / 2**8
plt.pcolormesh(
    X.transpose(),
    Y.transpose(),
    (elevation.transpose() > 0).astype(int) * humidity.transpose(),
)

# %%

biome = np.zeros((Xsize, Ysize, 3))
for i in range(Xsize):
    for j in range(Ysize):
        if elevation[i, j] < 0:
            biome[i, j] = hsv_to_rgb(0.7, 1, 0.6)
        else:
            biome[i, j] = np.square(np.cos(Y[i, j])) * (
                (np.log2(humidity[i, j] / 100) / 8 + 1)
                * np.array([0 / 255.0, 227 / 255.0, 174 / 255.0])
                + (1 - (np.log2(humidity[i, j] / 100) / 8 + 1))
                * np.array([227 / 255.0, 178 / 255.0, 0 / 255.0])
            ) + (1 - np.square(np.cos(Y[i, j]))) * np.array(
                [240 / 255.0, 240 / 255.0, 240 / 255.0]
            )

plt.figure()
plt.imshow(np.flip(np.swapaxes(biome, (0), (1)), axis=0))

# %%

# x = np.linspace(0, 1, 128).reshape((-1, 1))
# world1 = (1 - x) * np.array([[30.0 / 256, 33.0 / 256, 117.0 / 256, 1]]) + x * np.array(
#     [[52.0 / 256, 205.0 / 256, 235.0 / 256, 1]]
# )
# world2 = (1 - x) * np.array([[70.0 / 256, 235.0 / 256, 52.0 / 256, 1]]) + x * np.array(
#     [[235.0 / 256, 52.0 / 256, 52.0 / 256, 1]]
# )
# world = colors.LinearSegmentedColormap.from_list("world", np.vstack((world1, world2)))

# plt.pcolor(
#     X.transpose(),
#     Y.transpose(),
#     elevation.transpose(),
#     cmap=world,
# )
# cbar = plt.colorbar()
# plt.clim(-6.5, 6.5)

# plt.streamplot(longitudes, latitudes, VelX.transpose(), VelY.transpose(), density=2)
