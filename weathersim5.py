# %%

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
VelY = -0.2 * maxWind * np.sin(2 * Y * cellsPerHemisphere)

Humidity = np.square(np.cos(Y * cellsPerHemisphere))


# %%

surface = np.maximum(0, elevation)

dhdx = (np.roll(surface, -1, axis=0) - np.roll(surface, 1, axis=0)) / 2
dhdy = np.concatenate((surface[:, 1:], surface[:, -1:]), axis=1) - np.concatenate(
    (surface[:, :1], surface[:, :-1]), axis=1
)

DelVelX = -dhdy * np.sin(Y) / np.sqrt(1 + np.square(dhdx) + np.square(dhdy))
DelVelY = dhdx * np.sin(Y) / np.sqrt(1 + np.square(dhdx) + np.square(dhdy))

TotVelX = (VelX + 2 * surface / 6 * DelVelX) / (1 + surface / 6)
TotVelY = (VelY + 2 * surface / 6 * DelVelY) / (1 + surface / 6)

plt.pcolormesh(longitudes, latitudes, surface.transpose())
plt.streamplot(
    longitudes, latitudes, TotVelX.transpose(), TotVelY.transpose(), density=2
)

# %%


def treck(x, y):
    i, j = min(Xsize - 1, max(0, int(Xsize * x / (2 * np.pi)))), min(
        Ysize - 1, max(0, int(Ysize * (y / np.pi + 0.5)))
    )
    if elevation[i, j] < 0:
        return 0
    h = 0
    decay = 1
    # decay = amount = 0.95
    for i in range(200):
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
            h += 1 * np.square(np.cos(y)) * decay
        # amount *= decay
    return h


dynamicHumidity = np.vectorize(treck)(X, Y).astype(float)
dynamicHumidity *= 100.0 / dynamicHumidity.max()
print("Done")

# %%


filterSize = 200
variance = 1000
filter = (
    1
    / (2 * np.pi * variance)
    * np.exp(
        -1
        / (2 * variance)
        * np.sum(
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
water = (elevation < 0).astype(int) * np.square(np.cos(Y)) * Humidity
wrap = np.concatenate((water[-filterSize:], water, water[:filterSize]), axis=0)
wrap = np.concatenate(
    (
        wrap[::-1, 2 * filterSize - 1 :: -1],
        wrap,
        wrap[::-1, : -2 * filterSize - 1 : -1],
    ),
    axis=1,
)
# staticHumidity = (elevation >= 0).astype(int) * signal.convolve2d(filter, wrap, "valid")
staticHumidity = (elevation >= 0).astype(int) * signal.fftconvolve(
    wrap, filter, mode="valid"
)
lands = staticHumidity[(elevation >= 0)]
staticHumidity = (100 - 100.0 / 2**8) / (lands.max() - lands.min()) * (
    staticHumidity - lands.min()
) + 100.0 / 2**8
plt.pcolormesh(
    X.transpose(),
    Y.transpose(),
    (elevation.transpose() >= 0).astype(int) * staticHumidity.transpose(),
)

# %%

ratio = 0.5
totalhumidity = ratio * dynamicHumidity + (1 - ratio) * staticHumidity

biome = np.zeros((Xsize, Ysize, 3))
temps = np.zeros((Xsize, Ysize))
hums = np.zeros((Xsize, Ysize))
for i in range(Xsize):
    for j in range(Ysize):
        if elevation[i, j] < 0:
            biome[i, j] = hsv_to_rgb(0.7, 1, 0.6)
            temps[i, j] = -1
            hums[i, j] = -1
        else:
            temps[i, j] = int(
                max(
                    0,
                    min(5, 6 * (1 - abs(Y[i, j] / (np.pi / 2))) - elevation[i, j] / 6),
                )
            )
            temp = temps[i, j] / 5.0
            hums[i, j] = int(
                min(
                    2 + temps[i, j],
                    max(0, np.log2(totalhumidity[i, j] / 100) + 8),
                )
            )
            hum = hums[i, j] / (2 + temps[i, j])

            biome[i, j] = temp * (
                hum * np.array([0 / 255.0, 227 / 255.0, 174 / 255.0])
                + (1 - hum) * np.array([227 / 255.0, 178 / 255.0, 0 / 255.0])
            ) + (1 - temp) * np.array([240 / 255.0, 240 / 255.0, 240 / 255.0])

plt.figure()
plt.imshow(np.flip(np.swapaxes(biome, (0), (1)), axis=0))

# %%

for j in range(6):
    print("    " * (5 - j), end="")
    for i in range(3 + j):
        print(
            f" {np.count_nonzero(np.logical_and((temps == j), (hums==i)).astype(int)):6} ",
            end="",
        )
    print()

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
