from functools import cache
from typing import Optional, Self
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import toeplitz
from scipy.signal import convolve2d, gaussian

#%%

class Weather():
    df = pd.read_csv('assets/globe2.csv')
    scale = 5
    elevation = df.to_numpy()[:,1:]
    elevation = np.mean(elevation.reshape(elevation.shape[0], -1, scale), axis=2)
    elevation = np.mean(elevation.reshape(-1, scale, elevation.shape[1]), axis=1)
    land = np.array([[0,0,0,0], [0,0,0,0.1]])[(elevation > 0).astype(int)]

#%%

    radius = 10 * elevation.shape[1] / (2 * np.pi)
    angle = 23
    longitudes = np.linspace(0,360,elevation.shape[1],False)
    latitudes = 180 / np.pi * np.arcsin((2 * np.linspace(0,elevation.shape[0],elevation.shape[0],False) + 1) / elevation.shape[0] - 1)
    year = 120
    day = 0
    days = np.arange(year)

    T0 = 288        # K
    Tc = 400        # K
    dTdh = -5       # K/km
    dt = 3600       # s
    dx = scale      # km
    dy = dx * elevation.shape[1] / elevation.shape[0] / 2 # km
    dz = 12         # km

#%%

    D, Y = np.meshgrid(days, latitudes)
    a = 330**4
    b = 12

    @cache
    def daylight():
        declination = -np.arcsin(np.sin(Weather.Y * np.pi / 180) * np.sin(np.cos(2 * np.pi * Weather.D / Weather.year) * Weather.angle * np.pi / 180))
        return 24 / np.pi * np.arccos(np.minimum(np.maximum(np.tan(np.abs(Weather.Y) * np.pi / 180) * np.tan(declination), -1), 1))

    @cache
    def temperature(land: bool = True):
        # 0 = Weather.a * Weather.daylight(day) * np.cos(Weather.Y * np.pi / 180) - b * np.power(T, 4)
        raw = np.power(Weather.a * Weather.daylight() / Weather.b * np.cos(np.minimum(np.maximum((Weather.Y - np.cos(2 * np.pi * Weather.D / Weather.year) * Weather.angle), -90), 90) * np.pi / 180), 0.25)
        rate = 0.1
        transfer = toeplitz(np.concatenate([[1 - 2 * rate], [rate], np.zeros((998))]))
        transfer[0,0] = transfer[-1,-1] = 1 - rate
        transfer = np.linalg.matrix_power(transfer, 40000)
        
        blend = np.matmul(transfer, raw)
        if land:
            rate = 0.98
        else:
            rate = 0.98
        decay = np.exp(-(1 - rate) * np.arange(120))
        decay = decay / sum(decay)
        
        return np.array([np.convolve(np.concatenate([r, r[:len(decay) - 1]]), decay, 'valid') for r in blend])

    @cache
    def humidity():
        return np.square(np.cos(3 * Weather.Y * np.pi / 180)) * np.maximum(np.minimum(Weather.temperature(False) - 273, 30), 0) / 30

    @cache
    def thermalWind():
        rate = 0.5
        transfer = toeplitz(np.concatenate([[0], [rate], np.zeros((998))]), np.concatenate([[0], [-rate], np.zeros((998))]))
        transfer[0,0] = transfer[-1,-2] = -rate / 2
        transfer[0,1] = transfer[-1,-1] = rate / 2
        
        return -0*Weather.temperature(), np.matmul(transfer, Weather.temperature())

    @cache
    def frictionalWind():
        return -np.cos(Weather.Y * np.pi / 180), 0 * Weather.Y

    @cache
    def gravitationalWind():
        rate = 0.5
        transfer = toeplitz(np.concatenate([[0], [rate], np.zeros((998))]), np.concatenate([[0], [-rate], np.zeros((998))]))
        transfer[0,0] = transfer[-1,-2] = -rate / 2
        transfer[0,1] = transfer[-1,-1] = rate / 2
        
        return np.matmul(transfer, (Weather.elevation * (Weather.elevation > 0).astype(int))), \
            np.transpose(np.matmul(transfer, (Weather.elevation * (Weather.elevation > 0).astype(int)).transpose()))

#%%

    class Node:
        X: int
        Y: int
        Z: int
        Mesh: np.ndarray
        Thickness: float    # km
        Volume: float       # km^3
        Mass: float         # kg
        Temperature: float  # K
        Vx: float
        Vy: float
        
        density = 1         # kg/km^3
        
        cp = 1              # J/kg*K
        k = 1               # W/km*K
        a = density * cp / k# s/km^2
        sigma = 5.669e-2    # W/km^2K^4
        
        def __init__(self, x: int, y: int, z: int, mesh: np.ndarray,
                    thickness: Optional[float] = None, temperature: Optional[float] = None, mass: Optional[float] = None,
                    vx=0, vy=0) -> None:
            self.X = x
            self.Y = y
            self.Z = z
            self.Mesh = mesh
            
            if thickness:
                self.Thickness = thickness
            else:
                if self.Z < 1:
                    self.Thickness = Weather.dz + Weather.elevation[self.Y, self.X]
                elif self.Z == 1:
                    self.Thickness = -Weather.elevation[self.Y, self.X] * (Weather.elevation[self.Y, self.X] < 0).astype(int)
                else:
                    self.Thickness = Weather.dz - Weather.elevation[self.Y, self.X] * (Weather.elevation[self.Y, self.X] > 0).astype(int)
            
            if self.Z < 1:
                self.Volume = (2 * np.pi / mesh.shape[2]) \
                    * (2 / mesh.shape[1]) \
                    * (np.power(Weather.radius + Weather.elevation[self.Y, self.X], 3) - np.power(Weather.radius - Weather.dz, 3)) / 3
            elif self.Z == 1:
                self.Volume = (2 * np.pi / mesh.shape[2]) \
                    * (2 / mesh.shape[1]) \
                    * (np.power(Weather.radius, 3) - np.power(Weather.radius - Weather.elevation[self.Y, self.X] * (Weather.elevation[self.Y, self.X] < 0).astype(int), 3)) / 3
            else:
                self.Volume = (2 * np.pi / mesh.shape[2]) \
                    * (2 / mesh.shape[1]) \
                    * (np.power(Weather.radius + Weather.dz, 3) - np.power(Weather.radius - Weather.elevation[self.Y, self.X] * (Weather.elevation[self.Y, self.X] > 0).astype(int), 3)) / 3
            
            if temperature:
                self.Temperature = temperature
            else:
                self.Temperature = Weather.T0 + Weather.dTdh * Weather.elevation[self.Y, self.X] * (Weather.elevation > 0).astype(int)
            
            if mass:
                self.Mass = mass
            else:
                self.Mass = Weather.Node.density * self.Volume
                
            if vx:
                self.Vx = vx
            else:
                self.Vx = -np.cos(Weather.latitudes[self.Y]) * Weather.elevation[0].shape * Weather.scale / 24 / 3600
            
            if vy:
                self.Vy = vy
            else:
                self.Vy = 0
        
        def AreaX(self, y: int, z: int) -> float:
            if z < 1:
                return 0.5 * (np.power(Weather.radius + Weather.elevation[self.Y, self.X], 2) - np.power(Weather.radius - Weather.dz, 2)) \
                    * (2 / self.Mesh.shape[1])
            elif z == 1:
                return 0.5 * (np.power(Weather.radius + max(0, Weather.elevation[self.Y, self.X]), 2) - np.power(Weather.radius + Weather.elevation[self.Y, self.X], 2)) \
                    * (2 / self.Mesh.shape[1])
            else:
                return 0.5 * (np.power(Weather.radius + Weather.dz, 2) - np.power(Weather.radius + max(0, Weather.elevation[self.Y, self.X]), 2)) \
                    * (2 / self.Mesh.shape[1])

        def AreaY(self, y: int, z: int) -> float:
            if z < 1:
                return 0.5 * np.cos(np.arcsin(2 * y / self.Mesh.shape[2] - 1)) \
                    * (np.power(Weather.radius + Weather.elevation[self.Y, self.X], 2) - np.power(Weather.radius - Weather.dz, 2)) \
                    * (2 * np.pi / self.Mesh.shape[2])
            elif z == 1:
                return 0.5 * np.cos(np.arcsin(2 * y / self.Mesh.shape[2] - 1)) \
                    * (np.power(Weather.radius + max(0, Weather.elevation[self.Y, self.X]), 2) - np.power(Weather.radius + Weather.elevation[self.Y, self.X], 2)) \
                    * (2 * np.pi / self.Mesh.shape[2])
            else:
                return 0.5 * np.cos(np.arcsin(2 * y / self.Mesh.shape[2] - 1)) \
                    * (np.power(Weather.radius + Weather.dz, 2) - np.power(Weather.radius + max(0, Weather.elevation[self.Y, self.X]), 2)) \
                    * (2 * np.pi / self.Mesh.shape[2])
        
        def AreaZ(self, y: int, z: int) -> float:
            if z == 0:
                return np.power(Weather.radius - Weather.dz, 2) \
                    * (2 / self.Mesh.shape[1]) \
                    * (2 * np.pi / self.Mesh.shape[2])
            elif z == 1:
                return np.power(Weather.radius + Weather.elevation[self.Y, self.X], 2) \
                    * (2 / self.Mesh.shape[1]) \
                    * (2 * np.pi / self.Mesh.shape[2])
            elif z == 2:
                return np.power(Weather.radius + max(0, Weather.elevation[self.Y, self.X]), 2) \
                    * (2 / self.Mesh.shape[1]) \
                    * (2 * np.pi / self.Mesh.shape[2])
            else:
                return np.power(Weather.radius + Weather.dz, 2) \
                    * (2 / self.Mesh.shape[1]) \
                    * (2 * np.pi / self.Mesh.shape[2])

        def UpdateTemperature(self, intensity:float) -> float:
            temperature = self.Temperature
            
            if self.Thickness == 0:
                return temperature
            
            # Bottom
            if self.Z > 0:
                node = self.Mesh[self.Z-1, self.Y, self.X]
                if self.Z > 1 and node.Thickness == 0:
                    node =self.Mesh[self.Z-2, self.Y, self.X]
                temperature += 0.5 * (node.k + self.k) * self.AreaZ(self.Y, self.Z) * Weather.dt * (node.Temperature - self.Temperature) / (Weather.dz  * self.cp * self.Mass)
            
            # Top
            if self.Z < 2:
                node = self.Mesh[self.Z+1, self.Y, self.X]
                temperature += 0.5 * (self.k + node.k) * self.AreaZ(self.Y, self.Z+1) * Weather.dt * (node.Temperature - self.Temperature) / (Weather.dz  * self.cp * self.Mass)
                # Sunlight
                if (self.Z == 1 and self.Thickness > 0) or (self.Z == 0 and node.Thickness == 0):
                    temperature += self.AreaZ(self.Y, self.Z+1) * Weather.dt / (self.density * self.Volume * self.cp) * intensity / (self.cp * self.Mass)
                    temperature -= self.AreaZ(self.Y, self.Z+1) * Weather.dt * Weather.Node.sigma * np.power(self.Temperature, 4) / (self.cp * self.Mass)
            
            # North
            if self.Y < Weather.elevation.shape[0] - 1:
                node = self.Mesh[self.Z, self.Y+1, self.X]
                temperature += 0.5 * (self.k + node.k) * self.AreaY(self.Y+1, self.Z) * Weather.dt * (node.Temperature - self.Temperature) / (Weather.dy  * self.cp * self.Mass) \
                    - 0.5 * (self.k + node.k) * self.AreaY(self.Y+1, self.Z) * Weather.dt * self.Vy * (node.Temperature + self.Temperature) / 2
            
            # South
            if self.Y > 0:
                node = self.Mesh[self.Z, self.Y-1, self.X]
                temperature += 0.5 * (self.k + node.k) * self.AreaY(self.Y, self.Z) * Weather.dt * (node.Temperature - self.Temperature) / (Weather.dy  * self.cp * self.Mass)
            
            # East
            if self.X < Weather.elevation.shape[1] - 1:
                node = self.Mesh[self.Z, self.Y, self.X+1]
            else:
                node = self.Mesh[self.Z, self.Y, 0]
            temperature += 0.5 * (self.k + node.k) * self.AreaX(self.Y, self.Z) * Weather.dt * (node.Temperature - self.Temperature) / (Weather.dx  * self.cp * self.Mass)
            
            # West
            if self.X > 0:
                node = self.Mesh[self.Z, self.Y, self.X-1]
            else:
                node = self.Mesh[self.Z, self.Y, -1]
            temperature += 0.5 * (self.k + node.k) * self.AreaX(self.Y, self.Z) * Weather.dt * (node.Temperature - self.Temperature) / (Weather.dx  * self.cp * self.Mass)
                
            return temperature

        def UpdateVelocity(self):
            vx, vy = self.Vx, self.Vy
            
            return vx, vy

    class Earth(Node):
        
        density = 2.9e12    # kg/km^3
        
        cp = 800            # J/kg*K
        k = 1000            # W/km*K
        a = density * cp / k# s/km^2
        
        def __init__(self, x: int, y: int, z: int, mesh: np.ndarray,
                    thickness: Optional[float] = None, temperature: Optional[float] = None, mass: Optional[float] = None,
                    vx=0, vy=0) -> None:
            super().__init__(x, y, z, mesh, thickness, temperature, mass, vx, vy)
            
            if temperature:
                self.Temperature = temperature
            else:
                self.Temperature = Weather.T0 + Weather.dTdh * (Weather.elevation[self.Y, self.X] + Weather.dz) / 2
            
            if mass:
                self.Mass = mass
            else:
                self.Mass = Weather.Earth.density * self.Volume
                
        def Update(self, newMesh: np.ndarray, heat:float) -> Self:
            new = Weather.Earth(
                self.X,
                self.Y,
                self.Z,
                newMesh,
                self.Thickness,
                super().UpdateTemperature(heat),
                self.Mass
                *super().UpdateVelocity(),
            )
            newMesh[new.Z, new.Y, new.X] = new
            return new

    class Water(Node):
        
        density = 1e12      # kg/km^3
        
        cp = 4182           # J/kg*K
        k = 606             # W/km*K
        a = density * cp / k# s/km^2
        
        def __init__(self, x: int, y: int, z: int, mesh: np.ndarray,
                    thickness: Optional[float] = None, temperature: Optional[float] = None, mass: Optional[float] = None,
                    vx=0, vy=0) -> None:
            super().__init__(x, y, z, mesh, thickness, temperature, mass, vx, vy)
            
            if temperature:
                self.Temperature = temperature
            else:
                self.Temperature = Weather.T0 + Weather.dTdh * (-min(0, Weather.elevation[self.Y, self.X])) / 2
            
            if mass:
                self.Mass = mass
            else:
                self.Mass = Weather.Water.density * self.Volume

        def Update(self, newMesh: np.ndarray, heat:float) -> Self:
            new = Weather.Water(
                self.X,
                self.Y,
                self.Z,
                newMesh,
                self.Thickness,
                super().UpdateTemperature(heat),
                self.Mass
            )
            newMesh[new.Z, new.Y, new.X] = new
            return new
            
    class Air(Node):
        
        density = 1.2e9     # kg/km^3
        
        cp = 1005           # J/kg*K
        k = 26.2            # W/km*K
        a = density * cp / k# s/km^2
        
        def __init__(self, x: int, y: int, z: int, mesh: np.ndarray,
                    thickness: Optional[float] = None, temperature: Optional[float] = None, mass: Optional[float] = None,
                    vx=0, vy=0) -> None:
            super().__init__(x, y, z, mesh, thickness, temperature, mass, vx, vy)
            
            if temperature:
                self.Temperature = temperature
            else:
                self.Temperature = Weather.T0 + Weather.dTdh * (Weather.dz - Weather.elevation[self.Y, self.X]) / 2
            
            if mass:
                self.Mass = mass
            else:
                self.Mass = 1.2e9 * np.power(1.1, -(Weather.dz - Weather.elevation[self.Y, self.X]) / 2) * self.Volume

        def Update(self, newMesh: np.ndarray, heat:float) -> Self:
            new = Weather.Air(
                self.X,
                self.Y,
                self.Z,
                newMesh,
                self.Thickness,
                super().UpdateTemperature(heat),
                self.Mass
            )
            newMesh[new.Z, new.Y, new.X] = new
            return new
        
#%%

world = np.empty(dtype=Weather.Node, shape=(3, Weather.elevation.shape[0], Weather.elevation.shape[1]))

zs, ys, xs = np.meshgrid(np.arange(3), np.arange(Weather.elevation.shape[0]), np.arange(Weather.elevation.shape[1]))

def init(x: int, y: int, z: int):
    global world
    if z == 0:
        world[z, y, x] = Weather.Earth(x, y, z, world)
    elif z == 1:
        world[z, y, x] = Weather.Water(x, y, z, world)
    else:
        world[z, y, x] = Weather.Air(x, y, z, world)
init = np.vectorize(init)

init(xs, ys, zs)

#%%

score = np.inf
def update(frame: int):
    global world
    newWorld = np.empty(dtype=Weather.Node, shape=(3, Weather.elevation.shape[0], Weather.elevation.shape[1]))

    I0 = 1.4e9      # W/m^2
    def upd(x: int, y: int, z: int):
        newWorld[z, y, x] = world[z, y, x].Update(newWorld, I0)
    upd = np.vectorize(upd)
    upd(xs, ys, zs)
    
    def getTemp(node: Weather.Node):
        return node.Temperature
    getTemp = np.vectorize(getTemp)
    
    score = np.sqrt(np.mean(np.square(getTemp(world) - getTemp(newWorld))))
    print(frame, score)
    
    world = newWorld

for i in range(10):
    update(i)
