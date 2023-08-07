
from typing import Optional, Self
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d

#%%

df = pd.read_csv('myfirstlandia.csv')
elevation = df.to_numpy()[:,1:]
elevation = np.mean(elevation.reshape(elevation.shape[0], -1, 40), axis=2)
elevation = np.mean(elevation.reshape(-1, 40, elevation.shape[1]), axis=1)

T0 = 288        # K
Tc = 400        # K
dTdh = -5       # K/km
dt = 3600       # s
dx = 10*40      # km
dy = dx * elevation.shape[1] / elevation.shape[0] / 2 # km
dz = 1          # km

elevations = dz * np.linspace(-12, 12, 25)
radius = dx * elevation.shape[1] / (2 * np.pi)

#%%

class Node:
    X: int
    Y: int
    Z: int
    Mesh: np.ndarray
    Volume: float       # km^3
    Mass: float         # kg
    Temperature: float  # K
    Vx: float
    Vy: float
    Vz: float
    
    density = 1         # kg/km^3
    
    cp = 1              # J/kg*K
    k = 1               # W/km*K
    a = density * cp / k# s/km^2
    sigma = 5.669e-2    # W/km^2K^4
    
    def __init__(self, x: int, y: int, z: int, mesh: np.ndarray,
                 volume: Optional[float] = None, temperature: Optional[float] = None, mass: Optional[float] = None,
                 vx=0, vy=0, vz=0) -> None:
        self.X = x
        self.Y = y
        self.Z = z
        self.Mesh = mesh
            
        if volume:
            self.Volume = volume
        else:
            self.Volume = (2 * np.pi / mesh.shape[2]) \
                * (2 / mesh.shape[1]) \
                * (np.power(radius + elevations[z+1], 3) - np.power(radius + elevations[z], 3)) / 3
        
        if temperature:
            self.Temperature = temperature
        else:
            self.Temperature = T0 + dTdh * (elevations[z+1] + elevations[z]) / 2
        
        if mass:
            self.Mass = mass
        else:
            self.Mass = Node.density * self.Volume
            
        self.Vx = vx
        self.Vy = vy
        self.Vz = vz
    
    def AreaX(self, y: int, z: int) -> float:
        return 0.5 * (np.power(radius + elevations[z+1], 2) - np.power(radius + elevations[z], 2)) \
            * (2 / self.Mesh.shape[1])

    def AreaY(self, y: int, z: int) -> float:
        return 0.5 * np.cos(np.arcsin(2 * y / self.Mesh.shape[2] - 1)) \
            * (np.power(radius + elevations[z+1], 2) - np.power(radius + elevations[z], 2)) \
            * (2 * np.pi / self.Mesh.shape[2])
    
    def AreaZ(self, y: int, z: int) -> float:
        return np.power(radius + elevations[z], 2) \
            * (2 / self.Mesh.shape[1]) \
            * (2 * np.pi / self.Mesh.shape[2])

    def UpdateTemperature(self, intensity:float) -> float:
        temperature = self.Temperature
        
        # Bottom
        if self.Z == 0:
            temperature += 0.5 * (Earth.k + self.k) * self.AreaZ(self.Y, self.Z) * dt * (Tc - self.Temperature) / (dz  * self.cp * self.Mass)
        else:
            node = self.Mesh[self.Z-1, self.Y, self.X]
            temperature += 0.5 * (Earth.k + self.k) * self.AreaZ(self.Y, self.Z) * dt * (node.Temperature - self.Temperature) / (dz  * self.cp * self.Mass)
        
        # Top
        if self.Z < elevations.shape[0] - 2:
            node = self.Mesh[self.Z+1, self.Y, self.X]
            temperature += 0.5 * (self.k + node.k) * self.AreaZ(self.Y, self.Z+1) * dt * (node.Temperature - self.Temperature) / (dz  * self.cp * self.Mass)
            # Sunlight
            if self is not Air and node is Air:
                temperature += self.AreaZ(self.Y, self.Z+1) * dt / (self.density * self.Volume * self.cp) * intensity / (self.cp * self.Mass)
                temperature -= self.AreaZ(self.Y, self.Z+1) * dt * Node.sigma * np.power(self.Temperature, 4) / (self.cp * self.Mass)
        else:
            if self is not Air:
                temperature += self.AreaZ(self.Y, self.Z+1) * dt / (self.density * self.Volume * self.cp) * intensity / (self.cp * self.Mass)
                temperature -= self.AreaZ(self.Y, self.Z+1) * dt * Node.sigma * np.power(self.Temperature, 4) / (self.cp * self.Mass)
            
        
        # North
        if self.Y < elevation.shape[0] - 1:
            node = self.Mesh[self.Z, self.Y+1, self.X]
            temperature += 0.5 * (self.k + node.k) * self.AreaY(self.Y+1, self.Z) * dt * (node.Temperature - self.Temperature) / (dy  * self.cp * self.Mass)
        
        # South
        if self.Y > 0:
            node = self.Mesh[self.Z, self.Y-1, self.X]
            temperature += 0.5 * (self.k + node.k) * self.AreaY(self.Y, self.Z) * dt * (node.Temperature - self.Temperature) / (dy  * self.cp * self.Mass)
        
        # East
        if self.X < elevation.shape[1] - 1:
            node = self.Mesh[self.Z, self.Y, self.X+1]
            temperature += 0.5 * (self.k + node.k) * self.AreaX(self.Y, self.Z) * dt * (node.Temperature - self.Temperature) / (dx  * self.cp * self.Mass)
        else:
            node = self.Mesh[self.Z, self.Y, 0]
            temperature += 0.5 * (self.k + node.k) * self.AreaX(self.Y, self.Z) * dt * (node.Temperature - self.Temperature) / (dx  * self.cp * self.Mass)
        
        # West
        if self.X > 0:
            node = self.Mesh[self.Z, self.Y, self.X-1]
            temperature += 0.5 * (self.k + node.k) * self.AreaX(self.Y, self.Z) * dt * (node.Temperature - self.Temperature) / (dx  * self.cp * self.Mass)
        else:
            node = self.Mesh[self.Z, self.Y, -1]
            temperature += 0.5 * (self.k + node.k) * self.AreaX(self.Y, self.Z) * dt * (node.Temperature - self.Temperature) / (dx  * self.cp * self.Mass)
            
        
        return temperature

class Earth(Node):
    
    density = 2.9e12    # kg/km^3
    
    cp = 800            # J/kg*K
    k = 1000            # W/km*K
    a = density * cp / k# s/km^2
    
    def __init__(self, x: int, y: int, z: int, mesh: np.ndarray,
                 volume: Optional[float] = None, temperature: Optional[float] = None, mass: Optional[float] = None,
                 vx=0, vy=0, vz=0) -> None:
        super().__init__(x, y, z, mesh, volume, temperature, mass, vx, vy, vz)
        
        if temperature:
            self.Temperature = temperature
        else:
            self.Temperature = T0 + dTdh * (elevations[z+1] + elevations[z]) / 2
        
        if mass:
            self.Mass = mass
        else:
            self.Mass = Earth.density * self.Volume
            
    def Update(self, newMesh: np.ndarray, heat:float) -> Self:
        new = Earth(
            self.X,
            self.Y,
            self.Z,
            newMesh,
            self.Volume,
            super().UpdateTemperature(heat),
            self.Mass
        )
        newMesh[new.Z, new.Y, new.X] = new
        return new

class Water(Node):
    
    density = 1e12      # kg/km^3
    
    cp = 4182           # J/kg*K
    k = 606             # W/km*K
    a = density * cp / k# s/km^2
    
    def __init__(self, x: int, y: int, z: int, mesh: np.ndarray,
                 volume: Optional[float] = None, temperature: Optional[float] = None, mass: Optional[float] = None,
                 vx=0, vy=0, vz=0) -> None:
        super().__init__(x, y, z, mesh, volume, temperature, mass, vx, vy, vz)
        
        if temperature:
            self.Temperature = temperature
        else:
            self.Temperature = T0 + dTdh * (elevations[z+1] + elevations[z]) / 2
        
        if mass:
            self.Mass = mass
        else:
            self.Mass = Water.density * self.Volume

    def Update(self, newMesh: np.ndarray, heat:float) -> Self:
        new = Water(
            self.X,
            self.Y,
            self.Z,
            newMesh,
            self.Volume,
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
                 volume: Optional[float] = None, temperature: Optional[float] = None, mass: Optional[float] = None,
                 vx=0, vy=0, vz=0) -> None:
        super().__init__(x, y, z, mesh, volume, temperature, mass, vx, vy, vz)
        
        if temperature:
            self.Temperature = temperature
        else:
            self.Temperature = T0 + dTdh * (elevations[z+1] + elevations[z]) / 2
        
        if mass:
            self.Mass = mass
        else:
            self.Mass = 1.2e9 * np.power(1.1, -(elevations[z+1] + elevations[z]) / 2) * self.Volume

    def Update(self, newMesh: np.ndarray, heat:float) -> Self:
        new = Air(
            self.X,
            self.Y,
            self.Z,
            newMesh,
            self.Volume,
            super().UpdateTemperature(heat),
            self.Mass
        )
        newMesh[new.Z, new.Y, new.X] = new
        return new
        
#%%

world = np.empty(dtype=Node, shape=(elevations.shape[0]-1, elevation.shape[0], elevation.shape[1]))

for z in range(len(elevations)-1):
    temperature = T0 + dTdh * (elevations[z+1] + elevations[z]) / 2
    density = 1.2e9 * np.power(1.1, -(elevations[z+1] + elevations[z]) / 2)
    volume = (2 * np.pi / world.shape[2]) \
        * (2 / world.shape[1]) \
        * (np.power(radius + elevations[z+1], 3) - np.power(radius + elevations[z], 3)) / 3
    mass = density * volume
    for y in range(elevation.shape[0]):
        for x in range(elevation.shape[1]):
            if elevation[y, x] < elevations[z+1]:
                if elevations[z] < 0:
                    world[z, y, x] = Water(x, y, z, world, volume, temperature)
                else:
                    world[z, y, x] = Air(x, y, z, world, volume, temperature, mass)
            else:
                world[z, y, x] = Earth(x, y, z, world, volume, temperature)

#%%

v = np.vectorize(lambda n: n.Temperature)
a = np.maximum(-1, np.floor(elevation).astype(int)) + 12
b = np.take(np.moveaxis(world, 0, -1), a)

fig, ax = plt.subplots()
xdata, ydata = np.linspace(0, elevation.shape[1], elevation.shape[1]), np.linspace(0, elevation.shape[0], elevation.shape[0])
pc = ax.pcolorfast(v(b), cmap='coolwarm', vmin=200, vmax=400)
cbar = fig.colorbar(cm.ScalarMappable(norm=pc.norm, cmap=pc.cmap))

#%%

def update(frame):
    global world
    I0 = 1.4e9      # W/m^2
    
    def u(node: Node, mesh: np.ndarray, intensity: float):
        return node.Update(mesh, intensity)
    
    
    nextWorld = np.empty(dtype=Node, shape=world.shape)
    world = np.vectorize(u, excluded=['mesh', 'intensity'])(world, mesh=nextWorld, intensity=I0)
    
    v = np.vectorize(lambda n: n.Temperature)
    a = np.maximum(-1, np.floor(elevation).astype(int)) + 12
    b = np.take(np.moveaxis(world, 0, -1), a)
    pc.set_data(v(b))
    return pc,

# update(0)

#%%

ani = FuncAnimation(fig, update, frames=600,blit=True)

ani.save('basic_animation.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

#plt.show()
