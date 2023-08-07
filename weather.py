from functools import cache
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import toeplitz
from scipy.signal import convolve2d, gaussian

#%%

df = pd.read_csv('globe2.csv')
elevation = df.to_numpy()[:,1:]
land = np.array([[0,0,0,0], [0,0,0,0.1]])[(elevation > 0).astype(int)]

#%%

dx = 1
radius = dx * elevation.shape[1] / (2 * np.pi)
angle = 23
longitudes = np.linspace(0,360,1000,False)
latitudes = 180 / np.pi * np.arcsin((2 * np.linspace(0,1000,1000,False) + 1) / elevation.shape[0] - 1)
year = 120
day = 0
days = np.arange(year)

#%%

D, Y = np.meshgrid(days, latitudes)
a = 330**4
b = 12

@cache
def daylight():
    declination = -np.arcsin(np.sin(Y * np.pi / 180) * np.sin(np.cos(2 * np.pi * D / year) * angle * np.pi / 180))
    return 24 / np.pi * np.arccos(np.minimum(np.maximum(np.tan(np.abs(Y) * np.pi / 180) * np.tan(declination), -1), 1))

@cache
def temperature(land: bool = True):
    # 0 = a * daylight(day) * np.cos(Y * np.pi / 180) - b * np.power(T, 4)
    raw = np.power(a * daylight() / b * np.cos(np.minimum(np.maximum((Y - np.cos(2 * np.pi * D / year) * angle), -90), 90) * np.pi / 180), 0.25)
    rate = 0.1
    transfer = toeplitz(np.concatenate([[1 - 2 * rate], [rate], np.zeros((998))]))
    transfer[0,0] = transfer[-1,-1] = 1 - rate
    transfer = np.linalg.matrix_power(transfer, 40000)
    
    blend = np.matmul(transfer, raw)
    if land:
        rate = 0.98
    else:
        rate = 0.97
    decay = np.exp(-(1 - rate) * np.arange(120))
    decay = decay / sum(decay)
    
    return np.array([np.convolve(np.concatenate([r, r[:len(decay) - 1]]), decay, 'valid') for r in blend])

@cache
def humidity():
    return np.square(np.cos(3 * Y * np.pi / 180)) * np.maximum(np.minimum(temperature(False) - 273, 30), 0) / 30

@cache
def thermalWind():
    rate = 0.5
    transfer = toeplitz(np.concatenate([[0], [rate], np.zeros((998))]), np.concatenate([[0], [-rate], np.zeros((998))]))
    transfer[0,0] = transfer[-1,-2] = -rate / 2
    transfer[0,1] = transfer[-1,-1] = rate / 2
    
    return -0*temperature(), np.matmul(transfer, temperature())

@cache
def frictionalWind():
    return -np.cos(Y * np.pi / 180), 0 * Y

@cache
def gravitationalWind():
    rate = 0.5
    transfer = toeplitz(np.concatenate([[0], [rate], np.zeros((998))]), np.concatenate([[0], [-rate], np.zeros((998))]))
    transfer[0,0] = transfer[-1,-2] = -rate / 2
    transfer[0,1] = transfer[-1,-1] = rate / 2
    
    return np.matmul(transfer, (elevation * (elevation > 0).astype(int))), \
        np.transpose(np.matmul(transfer, (elevation * (elevation > 0).astype(int)).transpose()))

@cache
def wind():
    a = 0.1
    b = 0.1
    c = 0.1
    d = 1
    rate = 0.1
    mask = [[rate/2, rate, rate/2], [rate,1-rate*6,rate], [rate/2, rate, rate/2]]
    result = np.zeros((2, 120, 1000, 1000))
    for i in range(100):
        buffs = np.concatenate([result[:,:,1:2], result, result[:,:,-3:-2]],axis=2)
        print(i)
        result = a * np.repeat(np.moveaxis(np.array(thermalWind()), [1], [-1])[:,:,:,np.newaxis], 1000, axis=3) \
            + b * (np.repeat(np.moveaxis(np.array(frictionalWind()), [1], [-1])[:,:,:,np.newaxis], 1000, axis=3) - result) \
            + c * (np.repeat(np.array(gravitationalWind())[:,np.newaxis,:,:], year, axis=1)) \
            + d * [[convolve2d(np.concatenate([b[:,-1:], b, b[:,:1]],axis=1), mask, 'valid') for b in buff] for buff in buffs]
            
        
    return result

I, J = np.meshgrid(longitudes, latitudes)
R = 8.314       # J/K*mol
g = 0.029       # kg/mol

@cache
def wind(mass, dm_xdt, dm_ydt, temperature):
    '''
    dv/dt = grad(P) dA
    m/s^2 = kg*m/s^2*m^2/m
    
    d2m/dt2 = grad(P) * A
    kg/s^2 = kg*m/s^2*m^2*m * m^2
    
    d2m/dt2 = (dm/dr * R * T / g * V) + (m * R * dT/dr /g * V) * A
    
    P = m/g*R*T/V
    kg*m/s^2*m^2 = kg * mol/kg * kg*m/s^2*m/K*mol * K / m^3
    
    g = 0.029 kg/mol
    
    dm/dt / p / A = v
    '''
    
    d2m_xdt2 = (np.roll(mass, 1, axis=1) - np.roll(mass, -1, axis=1)) / (dx * 1000 / 360 * np.cos(J) * (np.roll(I, 1, axis=1) - np.roll(I, -1, axis=1))) * R * temperature / g * volume \
        + mass * R * (np.roll(temperature, 1, axis=1) - np.roll(temperature, -1, axis=1)) / (dx * 1000 / 360 * np.cos(J) * (np.roll(I, 1, axis=1) - np.roll(I, -1, axis=1))) / g * volume * areaX
    
    d2m_ydt2 = (np.concatenate([mass[0],mass[:-1]], axis=0) - np.concatenate([mass[1:],mass[-1]], axis=0)) / (dx * 1000 / 360 * (np.concatenate([J[0],J[:-1]], axis=0) - np.concatenate([J[1:],J[-1]], axis=0))) * R * temperature / g * volume \
        + mass * R * (np.concatenate([temperature[0],temperature[:-1]], axis=0) - np.concatenate([temperature[1:],temperature[-1]], axis=0)) / (dx * 1000 / 360 * (np.concatenate([J[0],J[:-1]], axis=0) - np.concatenate([J[1:],J[-1]], axis=0))) / g * volume * areaY
    
    
    

#%%

# # Save stuff
# np.save('world/elevation.npy', elevation)
# np.save('world/daylight.npy', daylight())
# np.save('world/temperature.npy', ((elevation > 0).astype(int)[np.newaxis,:,:] * (np.repeat(temperature().transpose()[:, :, np.newaxis], 1000, axis=2) + 6.5 * elevation[np.newaxis,:,:]) \
#     + (elevation <= 0).astype(int)[np.newaxis,:,:] * np.repeat(temperature(False).transpose()[:, :, np.newaxis], 1000, axis=2)))
# np.save('world/humidity.npy', ((elevation <= 0).astype(int) * np.repeat(humidity().transpose()[:, :, np.newaxis], 1000, axis=2)))
# np.save('world/wind.npy', np.moveaxis(wind(), [0], [-1]))

#%%

fig, ax = plt.subplots(2, 2, figsize=(12,6))
pc = ax[0,0].pcolormesh(days, latitudes, daylight(), cmap='plasma', vmin=0, vmax=24)
cbar = fig.colorbar(cm.ScalarMappable(cmap=pc.cmap, norm=pc.norm), ax=ax[0,0])
cbar.set_label('Sunlight (Hours)')
ax[0,0].set_title('Daylight (Hours)')
ax[0,0].set_xticks(np.linspace(0,120,5,True).astype(int))
ax[0,0].set_xlabel('Day ')
ax[0,0].set_yticks(np.linspace(-90,90,7,True).astype(int))
ax[0,0].set_ylabel('Latitude (°)')

pc = ax[0,1].pcolormesh(days, latitudes, temperature(), cmap='coolwarm', vmin=190, vmax=330)
cbar = fig.colorbar(cm.ScalarMappable(cmap=pc.cmap, norm=pc.norm), ax=ax[0,1])
cbar.set_label('Temperature (K)')
ax[0,1].set_title('Temperature')
ax[0,1].set_xticks(np.linspace(0,120,5,True).astype(int))
ax[0,1].set_xlabel('Day ')
ax[0,1].set_yticks(np.linspace(-90,90,7,True).astype(int))
ax[0,1].set_ylabel('Latitude (°)')

pc = ax[1,0].pcolormesh(days, latitudes, 100 * humidity(), cmap='coolwarm', vmin=0, vmax=100)
cbar = fig.colorbar(cm.ScalarMappable(cmap=pc.cmap, norm=pc.norm), ax=ax[1,0])
cbar.set_label('Humidity (%)')
ax[1,0].set_title('Humidty')
ax[1,0].set_xticks(np.linspace(0,120,5,True).astype(int))
ax[1,0].set_xlabel('Day ')
ax[1,0].set_yticks(np.linspace(-90,90,7,True).astype(int))
ax[1,0].set_ylabel('Latitude (°)')

pc = ax[1,1].quiver(D[::100,::3], Y[::100,::3], *[w[::100,::25] for w in gravitationalWind()], scale=10)
ax[1,1].set_title('Wind Currents')
ax[1,1].set_xticks(np.linspace(0,120,5,True).astype(int))
ax[1,1].set_xlabel('Day ')
ax[1,1].set_yticks(np.linspace(-90,90,7,True).astype(int))
ax[1,1].set_ylabel('Latitude (°)')

plt.tight_layout()

#%%

fig, ax = plt.subplots(2, 2, figsize=(12, 6))
plt.tight_layout()

def update(frame):
    pcs = []
    day = frame
    pc = ax[0,0].pcolormesh(longitudes, latitudes, np.repeat(daylight()[:, day:day+1], 1000, axis=1), cmap='plasma', vmin=0, vmax=24)
    pcs += [pc]
    cbar = fig.colorbar(cm.ScalarMappable(cmap=pc.cmap, norm=pc.norm), ax=ax[0,0])
    cbar.set_label('Sunlight (Hours)')
    ax[0,0].pcolormesh(longitudes, latitudes, land)
    ax[0,0].contour(longitudes, latitudes, elevation, levels=[-10,0,10], colors='black', linewidths=0.5)
    ax[0,0].set_title('Daylight Hours')
    ax[0,0].set_xticks(np.linspace(0,360,13,True).astype(int))
    ax[0,0].set_xlabel('Longitude (°)')
    ax[0,0].set_yticks(np.linspace(-90,90,7,True).astype(int))
    ax[0,0].set_ylabel('Latitude (°)')

    pc = ax[0,1].pcolormesh(longitudes, latitudes, 
                            (elevation > 0).astype(int) * (np.repeat(temperature()[:, day:day+1], 1000, axis=1) + 6.5 * elevation)
                            + (elevation <= 0).astype(int) * np.repeat(temperature(False)[:, day:day+1], 1000, axis=1), cmap='coolwarm', vmin=190, vmax=330)
    pcs += [pc]
    cbar = fig.colorbar(cm.ScalarMappable(cmap=pc.cmap, norm=pc.norm), ax=ax[0,1])
    cbar.set_label('Temperature (K)')
    ax[0,1].pcolormesh(longitudes, latitudes, land)
    ax[0,1].contour(longitudes, latitudes, elevation, levels=[-10,0,10], colors='black', linewidths=0.5)
    ax[0,1].set_title('Temperature')
    ax[0,1].set_xticks(np.linspace(0,360,13,True).astype(int))
    ax[0,1].set_xlabel('Longitude (°)')
    ax[0,1].set_yticks(np.linspace(-90,90,7,True).astype(int))
    ax[0,1].set_ylabel('Latitude (°)')

    pc = ax[1,0].pcolormesh(longitudes, latitudes, 
                            (elevation <= 0).astype(int) * np.repeat(100 * humidity()[:, day:day+1], 1000, axis=1), cmap='coolwarm', vmin=0, vmax=100)
    pcs += [pc]
    cbar = fig.colorbar(cm.ScalarMappable(cmap=pc.cmap, norm=pc.norm), ax=ax[1,0])
    cbar.set_label('Humidity (%)')
    ax[1,0].pcolormesh(longitudes, latitudes, land)
    ax[1,0].contour(longitudes, latitudes, elevation, levels=[-10,0,10], colors='black', linewidths=0.5)
    ax[1,0].set_title('Daylight Hours')
    ax[1,0].set_xticks(np.linspace(0,360,13,True).astype(int))
    ax[1,0].set_xlabel('Longitude (°)')
    ax[1,0].set_yticks(np.linspace(-90,90,7,True).astype(int))
    ax[1,0].set_ylabel('Latitude (°)')

    pc = ax[1,1].quiver(3 * D[::100,::3], Y[::100,::3], *[w[day,::100,::25] for w in wind()], scale=10)
    pcs += [pc]
    ax[1,1].pcolormesh(longitudes, latitudes, land)
    ax[1,1].contour(longitudes, latitudes, elevation, levels=[-10,0,10], colors='black', linewidths=0.5)
    ax[1,1].set_title('Wind Currents')
    ax[1,1].set_xticks(np.linspace(0,360,13,True).astype(int))
    ax[1,1].set_xlabel('Day ')
    ax[1,1].set_yticks(np.linspace(-90,90,7,True).astype(int))
    ax[1,1].set_ylabel('Latitude (°)')
    
    return pcs

update(0)

#%%

# ani = FuncAnimation(fig, update, frames=120,blit=True)

# ani.save('weather.mp4', fps=12, extra_args=['-vcodec', 'libx264'])