import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
Xsize = 1000
Ysize = 500
radius = Xsize / (2 * np.pi)
w = 1 * 2 * np.pi / (24 * 3600)
atmosphere = 20
surfacePressure = 1e5
g = -9.8e-3
M = 0.02896
R = 8.314e-6
T0 = 290

longitudes = np.linspace(0, 2 * np.pi, 1000, False)
latitudes = np.linspace(
    -np.pi / 2 * ((2 * Ysize - 1) / (2 * Ysize)),
    np.pi / 2 * ((2 * Ysize - 1) / (2 * Ysize)),
    Ysize,
    True,
)
altitudes = atmosphere * np.linspace(0, 1, 3, True)

X, Y, Z = np.meshgrid(longitudes, latitudes, altitudes, indexing="ij")

# Ay =
Az = np.power(radius + Z, 2) * (2 / Ysize) * (2 * np.pi / Xsize)

# %%

sunlight = 1.0 / 60 * np.cos(Y) * (1 - Z / altitudes)
cooling = -1.0 / 60 * (1 - np.cos(Y)) * Z / altitudes

# %%

"""
Vertically, gravitational pressure is the dominating force
This is modeled by
P_g = P0*e^(M0*g*h/R0*T)
Heat creates a pressure differential, thereby expanding the mass distribution upwards
This mass distribution differential creates a pressure differential by the ideal gas law
PV = (m/M)*R*T
Where less mass in lower cells results in lower ambient pressure and higher mass -> higher pressure
This differential in the horizontal plane results in wind
du/dt = -dP/dx / p
p = m/V = P*M/R*T
dP/dx = p*R*dT/dx/M + -M0*g*h/R0*T^2*dt/dx *P0*e^(M0*g*h/R0*T)
du/dt = -R*dT/dx/M + g*h/T*dt/dx

dw/dt = -dP/dz / p
dP/dz = -(P - P0) / (h - h0)
dw/dt = -(1 - P0*e^(M0*g*h0/R0*T0) / P0*e^(M0*g*h/R0*T)) / (h - h0) / M/R*T
"""

# Temperature = T0 + 0 * np.cos(Y)
# Volume = (
#     (2 * np.pi / Xsize)
#     * (2 / Ysize)
#     * (np.power(radius + Z[:, :, 1:], 3) - np.power(radius + Z[:, :, :-1], 3))
#     / 3
# )
# Mass = (
#     Az[:, :, :-1]
#     * -surfacePressure
#     * (
#         np.exp(M * g * Z[:, :, :-1] / R / Temperature[:, :, :-1])
#         - np.exp(M * g * Z[:, :, 1:] / R / Temperature[:, :, :-1])
#     )
#     / g
# )
# VelocityX = 0 * Z[:, :, :-1]
# VelocityY = 0 * Z[:, :, :-1]

# %%

Xsize = 40000
Ysize = 20000
radius = Xsize / (2 * np.pi)
w = 1 * 2 * np.pi / (24 * 3600)
atmosphere = 20
surfacePressure = 1e5
g = -9.8e-3
M = 0.02896
R = 8.314e-6
T0 = 290

elements = int(1e3)
PosX = np.zeros((elements,))
PosY = radius * np.arcsin(np.random.uniform(-1, 1, size=(elements,)))
Temp = T0 + 3 * np.cos(2 * PosY / radius)
PosZ = R * Temp / (M * g) * np.log(1 - np.random.uniform(0, 1, size=(elements,)))
VelX = np.zeros((elements,))
VelY = np.random.uniform(-0.001, 0.001, (elements,))
VelZ = np.zeros((elements,))

dt = 1
for i in range(24 * 60 * 60):
    f = 2 * w * np.sin(PosY / radius)
    T = T0 + 3 * np.cos(2 * PosY / radius)
    dTdy = -3 * 2 / radius * np.sin(2 * PosY / radius)
    VelZ = np.power(0.5, dt) * VelZ + (-R * Temp / (M * g) - PosZ) * 1e-3 * dt
    Temp += 0 * (T - Temp) * np.exp(-PosZ) * 1e-9 * dt
    PosX += VelX * dt / np.cos(PosY / radius)
    PosY += VelY * dt
    PosZ += VelZ * dt
    ax = VelY * f
    ay = -VelX * f + (R * dTdy / M + g * PosZ / T * dTdy)
    VelX += ax * dt
    VelY += ay * dt

    if i == 0:
        plt.figure()
        plt.quiver(PosY, PosZ, VelY, VelZ)
        plt.show()

plt.quiver(PosY, PosZ, VelY, VelZ)

# %%

elements = int(1e5)
PosX = np.zeros((elements,))
PosY = 0 * radius * np.arcsin(np.random.uniform(-1, 1, size=(elements,)))
Temp = T0 + 30 * np.cos(2 * PosY / radius)
PosZ = 10 + 0 * R * Temp / (M * g) * np.log(
    1 - np.random.uniform(0, 1, size=(elements,))
)
VelX = np.zeros((elements,))
VelY = np.zeros((elements,)) + 0.01
VelZ = np.zeros((elements,))
AccX = np.zeros((elements,))
AccY = np.zeros((elements,))
AccY1 = np.zeros((elements,))
AccY2 = np.zeros((elements,))

dt = 1
for i in range(1, elements):
    f = 2 * w * np.sin(PosY[i - 1] / radius)
    T = T0 + 3 * np.cos(2 * PosY[i - 1] / radius)
    dTdy = -3 * 2 / radius * np.sin(2 * PosY[i - 1] / radius)
    VelZ[i] = (
        np.power(0.90, dt) * VelZ[i - 1]
        + (-R * Temp[i - 1] / (M * g) - PosZ[i - 1]) * 1e-3 * dt
    )
    Temp[i] = Temp[i - 1] + (T - Temp[i - 1]) * np.exp(-PosZ[i - 1]) * 1e-0 * dt
    PosX[i] = PosX[i - 1] + VelX[i - 1] * dt / np.cos(PosY[i - 1] / radius)
    PosY[i] = PosY[i - 1] + VelY[i - 1] * dt
    PosZ[i] = PosZ[i - 1] + VelZ[i - 1] * dt
    AccX[i - 1] = VelY[i - 1] * f
    AccY1[i - 1] = -VelX[i - 1] * f
    AccY2[i - 1] = R * dTdy / M + g * PosZ[i - 1] / T * dTdy
    AccY[i - 1] = AccY1[i - 1] + AccY2[i - 1]
    VelX[i] = VelX[i - 1] + AccX[i - 1] * dt
    VelY[i] = VelY[i - 1] + AccY[i - 1] * dt

plt.plot(PosY, PosZ)

# %%

"""
p = m/V
PV = m/M * R*T
p = m/V = P*M/R*T
p*R*T/M = P
m = P*V*M/R*T

P = R/M*p(h)*T(h) + S P*M/R*T * g*dh = Spgdh
dP/dh = P*M/R*T * g
m = SdP/dh*dV = S P*M/R*T * g * dA(h) * dh
P = P0*e^(M*g*h/R*T)
dw/dt = -dP/dh/p + g
"""

# %%

vx = 0
vy = 0.5 * 0.1 / 60 / 60
px = 0
py = 0 * np.pi / 6 * radius
pxs = [px]
pys = [py]
dt = 1
for i in range(1000000):
    f = 2 * w * np.sin(py)
    px += vx * dt / np.cos(py)
    py += vy * dt
    vx += vy * dt * f
    vy -= vx * dt * f
    pxs += [px]
    pys += [py]

plt.plot(pxs, pys)
