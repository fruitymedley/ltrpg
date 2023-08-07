from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as interp

#%%

df = pd.read_csv('globe2.csv')
elevation = df.to_numpy()[:,1:]

#%%

x = np.linspace(0, 1, 128).reshape((-1,1))
world1 = (1 - x) * np.array([[30.0 / 256, 33.0 / 256, 117.0 / 256, 1]]) + x * np.array([[52.0 / 256, 205.0 / 256, 235.0 / 256, 1]])
world2 = (1 - x) * np.array([[70.0 / 256, 235.0 / 256, 52.0 / 256, 1]]) + x * np.array([[235.0 / 256, 52.0 / 256, 52.0 / 256, 1]])
world = colors.LinearSegmentedColormap.from_list('world', np.vstack((world1, world2)))
c = world(colors.Normalize(-6.5,6.5)(elevation))

#%%

Is = np.linspace(0, 1000, elevation.shape[1], False) + 0.5
Js = np.linspace(0, 1000, elevation.shape[0], False)

#%%

longitudes = 2 * np.pi * Is / elevation.shape[1]
latitudes = np.arcsin((2 * Js + 1) / elevation.shape[0] - 1)
eFunc = interp([latitudes, longitudes], elevation, bounds_error=False, fill_value=None)

#%%

xs = np.cos(longitudes) * np.cos(latitudes)
ys = np.sin(latitudes)
zs = np.sin(longitudes) * np.cos(latitudes)

#%%

size = int(elevation.shape[1] / np.pi)
xdata, ydata = np.linspace(-1+1.0/size, 1-1.0/size, size), np.linspace(-1+1.0/size, 1-1.0/size, size)
xs, ys = np.meshgrid(xdata, ydata)

def pixel(x, y, frame):
    if np.sqrt(np.power(x, 2) + np.power(y, 2)) <= 1:
        return eFunc([np.arcsin(y), ((np.arcsin(x / np.cos(np.arcsin(y))) + np.pi + 2 * np.pi * frame / 100) % (2 * np.pi))])[0]
    return np.nan
v = np.vectorize(pixel)

def render(frame):
    global v
            
    return v(xs, ys, frame)

#%%

fig, ax = plt.subplots()
pc = ax.pcolorfast(render(50), cmap=world, vmin=-6.5, vmax=6.5)
ax.axis('off')
cbar = fig.colorbar(cm.ScalarMappable(cmap=world, norm=pc.norm))

#%%

def update(frame):
    print(frame)
    pc.set_data(render(frame))
    return pc,

#%%

ani = FuncAnimation(fig, update, frames=100, blit=True)

ani.save('globe2.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

