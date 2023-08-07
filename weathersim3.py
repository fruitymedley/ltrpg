import numpy as np
import pandas as pd

# %%

df = pd.read_csv("assets/globe2.csv")
elevation = df.to_numpy()[:, 1:]

# %%

R = 8.314  # J/K*mol
g = 0.029  # kg/mol
dt = 1
I0 = 1.4e9

# %%


class World:
    SizeX: int
    SizeY: int
    SizeZ: int
    Scale: float
    Radius: float
    Tilt: float

    LongitudesNodes: np.ndarray
    LatitudesNodes: np.ndarray

    Xs: np.ndarray
    Ys: np.ndarray
    Zs: np.ndarray

    AzimuthsNodes: np.ndarray
    AltitudesNodes: np.ndarray
    ElevationsNodes: np.ndarray

    AzimuthsFaces: np.ndarray
    AltitudesFaces: np.ndarray
    ElevationsXFaces: np.ndarray
    ElevationsYFaces: np.ndarray
    ElevationsFaces: np.ndarray

    Temperature: np.ndarray
    Volume: np.ndarray
    Mass: np.ndarray
    VelocityX: np.ndarray
    VelocityY: np.ndarray
    AreaX: np.ndarray
    AreaY: np.ndarray
    AreaZ: np.ndarray

    earthDensity = 2.9e12  # kg/km^3
    waterDensity = 1e12  # kg/km^3
    airDensity = 1.2e9  # kg/km^3

    earthK = 1000  # W/km*K
    waterK = 606  # W/km*K
    airK = 26.2  # W/km*K

    earthCp = 800  # J/kg*K
    waterCp = 4182  # J/kg*K
    airCp = 1005  # J/kg*K

    sigma = 5.669e-2  # W/km^2K^4

    def __init__(
        self,
        elevation: np.ndarray,
        scale: float = 1,
        depth: float = 12,
        tilt: float = 23,
        lenYear: int = 365,
        lenDay: int = 24,
        day: int = 0,
    ) -> None:
        self.SizeX = elevation.shape[1]
        self.SizeY = elevation.shape[0]
        self.SizeZ = 4

        self.Scale = scale
        self.Radius = self.Scale * self.SizeX / (2 * np.pi)
        self.Depth = depth
        self.Tilt = tilt
        self.LongitudesNodes = np.linspace(
            np.pi / self.SizeX, 2 * np.pi - np.pi / self.SizeX, self.SizeX, True
        )
        self.LatitudesNodes = np.arcsin(
            np.linspace(-1 + 1 / self.SizeY, 1 - 1 / self.SizeY, self.SizeY, True)
        )
        self.LongitudesFaces = np.linspace(
            0, 2 * np.pi * (1 - 1 / self.SizeX), self.SizeX, True
        )
        self.LatitudesFaces = np.arcsin(np.linspace(-1, 1, self.SizeY + 1, True))

        self.LenYear = lenYear
        self.LenDay = lenDay
        self.Day = day

        self.Xs, self.Ys, self.Zs = np.meshgrid(
            np.arange(self.SizeX), np.arange(self.SizeY), np.arange(self.SizeZ)
        )
        self.AzimuthsNodes, self.AltitudesNodes, self.ElevationsNodes = np.meshgrid(
            self.LongitudesNodes,
            self.LatitudesNodes,
            [-self.Depth, 0, self.Depth],
            indexing="ij",
        )
        self.ElevationsNodes[:, :, 0] = (-self.Depth + elevation) / 2
        self.ElevationsNodes[:, :, 1] = (elevation + np.maximum(0, elevation)) / 2
        self.ElevationsNodes[:, :, 2] = (np.maximum(0, elevation) + self.Depth) / 2

        self.AzimuthsFaces, _, _ = np.meshgrid(
            self.LongitudesFaces,
            self.LatitudesNodes,
            [-self.Depth, 0, self.Depth],
            indexing="ij",
        )
        _, self.AltitudesFaces, _ = np.meshgrid(
            self.LongitudesNodes,
            self.LatitudesFaces,
            [-self.Depth, 0, self.Depth],
            indexing="ij",
        )
        _, _, self.ElevationsFaces = np.meshgrid(
            self.LongitudesNodes,
            self.LatitudesNodes,
            [-self.Depth, 0, 0, self.Depth],
            indexing="ij",
        )
        self.ElevationsFaces[:, :, 1] = elevation
        self.ElevationsFaces[:, :, 2] = np.maximum(0, elevation)
        self.ElevationsXFaces = (
            self.ElevationsFaces + np.roll(self.ElevationsFaces, -1, axis=0)
        ) / 2
        self.ElevationsYFaces = np.concatenate(
            [
                np.repeat(
                    self.ElevationsFaces[:, 0:1].mean(axis=0)[np.newaxis],
                    self.SizeX,
                    axis=0,
                ),
                (self.ElevationsFaces[:, 1:] + self.ElevationsFaces[:, :-1]) / 2,
                np.repeat(
                    self.ElevationsFaces[:, -2:-1].mean(axis=0)[np.newaxis],
                    self.SizeX,
                    axis=0,
                ),
            ],
            axis=1,
        )

        self.Temperature = 290 + 40 * np.cos(self.AltitudesNodes)
        self.Volume = (
            (2 * np.pi / self.SizeX)
            * (2 / self.SizeY)
            * (
                np.power(self.Radius + self.ElevationsFaces[1:], 3)
                - np.power(self.Radius + self.ElevationsFaces[:-1], 3)
            )
            / 3
        )
        self.Mass = np.copy(self.Volume)
        self.Mass[:, :, 0] *= World.earthDensity
        self.Mass[:, :, 1] *= World.waterDensity
        self.Mass[:, :, 2] *= World.airDensity
        with np.errstate(divide="ignore", invalid="ignore"):
            self.VelocityX = np.nan_to_num(
                2
                * np.pi
                / (self.LenDay * 3600)
                / (self.ElevationsXFaces[:, :, 1:] - self.ElevationsXFaces[:, :, :-1])
                * 0.5
                * (
                    np.square(self.Radius + self.ElevationsXFaces[:, :, 1:])
                    - np.square(self.Radius + self.ElevationsXFaces[:, :, :-1])
                )
            )
        self.VelocityY = np.zeros((self.SizeX, self.SizeY + 1, self.SizeZ - 1))

        self.AreaX = (
            0.5
            * (
                np.power(self.Radius + self.ElevationsXFaces[:, :, 1:], 2)
                - np.power(self.Radius + self.ElevationsXFaces[:, :, :-1], 2)
            )
            * (2 / self.SizeY)
        )
        self.AreaY = (
            0.5
            * np.cos(self.AltitudesFaces)
            * (
                np.power(self.Radius + self.ElevationsYFaces[:, :, 1:], 2)
                - np.power(self.Radius + self.ElevationsYFaces[:, :, :-1], 2)
            )
            * (2 * np.pi / self.SizeX)
        )
        self.AreaZ = (
            np.power(self.Radius + self.ElevationsFaces, 2)
            * (2 / self.SizeY)
            * (2 * np.pi / self.SizeX)
        )

    def Update(self, intensity) -> None:
        temperature = np.copy(self.Temperature)
        mass = np.copy(self.Mass)
        velocityX = np.copy(self.VelocityX)
        velocityY = np.copy(self.VelocityY)

        # Earth
        temperature[:, :, 0] += (
            (
                World.earthK
                * self.AreaX[:, :, 0]
                * dt
                * (
                    np.roll(self.Temperature, -1, axis=0)[:, :, 0]
                    - self.Temperature[:, :, 0]
                )
                / (
                    self.Radius
                    * np.cos(self.LatitudesNodes)
                    * (
                        self.LongitudesNodes - np.roll(self.LongitudesNodes, -1, axis=0)
                    )[:, :, 0]
                    * World.earthCp
                    * self.Mass[:, :, 0]
                )
            )
            + (
                World.earthK
                * np.roll(self.AreaX[:, :, 0], 1, axis=0)
                * dt
                * (
                    np.roll(self.Temperature, 1, axis=0)[:, :, 0]
                    - self.Temperature[:, :, 0]
                )
                / (
                    self.Radius
                    * np.cos(self.LatitudesNodes)
                    * (np.roll(self.LongitudesNodes, 1, axis=0) - self.LongitudesNodes)[
                        :, :, 0
                    ]
                    * World.earthCp
                    * self.Mass[:, :, 0]
                )
            )
            + (
                World.earthK
                * self.AreaY[:, :-1, 0]
                * dt
                * (
                    np.roll(self.Temperature, -1, axis=1)[:, :, 0]
                    - self.Temperature[:, :, 0]
                )
                / (
                    self.Radius
                    * (np.roll(self.LatitudesNodes, 1, axis=0) - self.LatitudesNodes)[
                        :, :, 0
                    ]
                    * World.earthCp
                    * self.Mass[:, :, 0]
                )
            )
            + (
                World.earthK
                * self.AreaY[:, 1:, 0]
                * dt
                * (
                    np.roll(self.Temperature, 1, axis=1)[:, :, 0]
                    - self.Temperature[:, :, 0]
                )
                / (
                    self.Radius
                    * (np.roll(self.LatitudesNodes, 1, axis=0) - self.LatitudesNodes)[
                        :, :, 0
                    ]
                    * World.earthCp
                    * self.Mass[:, :, 0]
                )
            )
            + (
                (self.Volume[:, :, 1] > 0).astype(int)
                * (
                    (np.roll(self.ElevationsNodes, 1, axis=2) - self.ElevationsNodes)[
                        :, :, 0
                    ]
                    * World.earthK
                    + (
                        np.roll(self.ElevationsNodes, 2, axis=2)
                        - np.roll(self.ElevationsNodes, 1, axis=2)
                    )[:, :, 0]
                    * World.waterK
                )
                / (np.roll(self.ElevationsNodes, 2, axis=2) - self.ElevationsNodes)
                * self.AreaZ[:, :, 1]
                * dt
                * (
                    np.roll(self.Temperature, 1, axis=2)[:, :, 0]
                    - self.Temperature[:, :, 0]
                )
                / (
                    self.Radius
                    * (np.roll(self.ElevationsNodes, 1, axis=2) - self.ElevationsNodes)[
                        :, :, 0
                    ]
                    * World.earthCp
                    * self.Mass[:, :, 0]
                )
            )
            + (
                (self.Volume[:, :, 1] == 0).astype(int)
                * (
                    (np.roll(self.ElevationsNodes, 1, axis=2) - self.ElevationsNodes)[
                        :, :, 0
                    ]
                    * World.earthK
                    + (
                        np.roll(self.ElevationsNodes, 3, axis=2)
                        - np.roll(self.ElevationsNodes, 2, axis=2)
                    )[:, :, 0]
                    * World.airK
                )
                / (np.roll(self.ElevationsNodes, 3, axis=2) - self.ElevationsNodes)
                * self.AreaZ[:, :, 2]
                * dt
                * (
                    np.roll(self.Temperature, 2, axis=2)[:, :, 0]
                    - self.Temperature[:, :, 0]
                )
                / (
                    self.Radius
                    * (np.roll(self.ElevationsNodes, 2, axis=2) - self.ElevationsNodes)[
                        :, :, 0
                    ]
                    * World.earthCp
                    * self.Mass[:, :, 0]
                )
            )
            + (
                (self.Volume[:, :, 1] == 0).astype(int)
                * self.AreaZ[:, :, 1]
                * dt
                / (self.Mass * self.earthCp)
                * intensity
            )
            - (
                (self.Volume[:, :, 1] == 0).astype(int)
                * self.AreaZ[:, :, 1]
                * dt
                * World.sigma
                * np.power(self.Temperature[:, :, 0], 4)
                / (self.Mass[:, :, 0] * self.earthCp)
            )
        )

        # Water
        with np.errstate(divide="ignore", invalid="ignore"):
            temperature[:, :, 1] += np.nan_to_num(
                World.waterK
                * self.AreaX[:, :, 1]
                * dt
                * (
                    np.roll(self.Temperature, -1, axis=0)[:, :, 1]
                    - self.Temperature[:, :, 1]
                )
                / (
                    self.Radius
                    * np.cos(self.LatitudesNodes)
                    * (
                        self.LongitudesNodes - np.roll(self.LongitudesNodes, -1, axis=0)
                    )[:, :, 1]
                    * World.waterCp
                    * self.Mass[:, :, 1]
                )
                + World.waterK
                * np.roll(self.AreaX[:, :, 1], 1, axis=0)
                * dt
                * (
                    np.roll(self.Temperature, 1, axis=0)[:, :, 1]
                    - self.Temperature[:, :, 1]
                )
                / (
                    self.Radius
                    * np.cos(self.LatitudesNodes)
                    * (np.roll(self.LongitudesNodes, 1, axis=0) - self.LongitudesNodes)[
                        :, :, 1
                    ]
                    * World.waterCp
                    * self.Mass[:, :, 1]
                )
                + World.waterK
                * self.AreaY[:, :-1, 1]
                * dt
                * (
                    np.roll(self.Temperature, -1, axis=1)[:, :, 1]
                    - self.Temperature[:, :, 1]
                )
                / (
                    self.Radius
                    * (np.roll(self.LatitudesNodes, 1, axis=0) - self.LatitudesNodes)[
                        :, :, 1
                    ]
                    * World.waterCp
                    * self.Mass[:, :, 1]
                )
                + World.waterK
                * self.AreaY[:, 1:, 1]
                * dt
                * (
                    np.roll(self.Temperature, 1, axis=1)[:, :, 1]
                    - self.Temperature[:, :, 1]
                )
                / (
                    self.Radius
                    * (np.roll(self.LatitudesNodes, 1, axis=0) - self.LatitudesNodes)[
                        :, :, 1
                    ]
                    * World.waterCp
                    * self.Mass[:, :, 1]
                )
                + (
                    (np.roll(self.ElevationsNodes, 1, axis=2) - self.ElevationsNodes)[
                        :, :, 0
                    ]
                    * World.earthK
                    + (
                        np.roll(self.ElevationsNodes, 2, axis=2)
                        - np.roll(self.ElevationsNodes, 1, axis=2)
                    )[:, :, 0]
                    * World.waterK
                )
                / (np.roll(self.ElevationsNodes, 2, axis=2) - self.ElevationsNodes)
                * self.AreaZ[:, :, 1]
                * dt
                * (
                    np.roll(self.Temperature, 1, axis=2)[:, :, 0]
                    - self.Temperature[:, :, 0]
                )
                / (
                    self.Radius
                    * (np.roll(self.ElevationsNodes, 1, axis=2) - self.ElevationsNodes)[
                        :, :, 0
                    ]
                    * World.waterCp
                    * self.Mass[:, :, 0]
                )
                + (
                    (np.roll(self.ElevationsNodes, 2, axis=2) - self.ElevationsNodes)[
                        :, :, 0
                    ]
                    * World.waterK
                    + (
                        np.roll(self.ElevationsNodes, 3, axis=2)
                        - np.roll(self.ElevationsNodes, 2, axis=2)
                    )[:, :, 0]
                    * World.airK
                )
                / (np.roll(self.ElevationsNodes, 3, axis=2) - self.ElevationsNodes)
                * self.AreaZ[:, :, 2]
                * dt
                * (
                    np.roll(self.Temperature, 2, axis=2)[:, :, 0]
                    - self.Temperature[:, :, 0]
                )
                / (
                    self.Radius
                    * (np.roll(self.ElevationsNodes, 2, axis=2) - self.ElevationsNodes)[
                        :, :, 0
                    ]
                    * World.waterCp
                    * self.Mass[:, :, 0]
                )
                + self.AreaZ[:, :, 2] * dt / (self.Mass * self.earthCp) * intensity
                - self.AreaZ[:, :, 2]
                * dt
                * World.sigma
                * np.power(self.Temperature[:, :, 1], 4)
                / (self.Mass[:, :, 1] * self.waterCp)
            )
            # v_x = dP/dx / p
            # dP/dx = (P_x+1 - P_x-1) / dx = 1/2p((v_x-1)^2 - (v_x+1)^2)
            #
            velocityX[:, :, 1] += (
                # Filters
                # Between water
                (self.Volume[:, :, 1] > 0).astype(int)
                * (np.roll(self.Volume[:, :, 1], 1, axis=1) > 0).astype(int)
                *
                # Gradient of pressure
                0.5
                * (
                    # v_1^2
                    np.square(
                        (
                            np.roll(self.VelocityX[:, :, 1], 1, axis=0)
                            + self.VelocityX[:, :, 1]
                        )
                        / 2
                    )
                    + np.square(
                        (
                            np.roll(self.VelocityY[:, :-1, 1], 1, axis=0)
                            + np.roll(self.VelocityY[:, 1:, 1], 1, axis=0)
                        )
                        / 2
                    )
                    -
                    # v2^2
                    np.square(
                        (
                            self.VelocityX[:, :, 1]
                            + np.roll(self.VelocityX[:, :, 1], -1, axis=0)
                        )
                        / 2
                    )
                    - np.square(
                        (
                            np.roll(self.VelocityY[:, :-1, 1], 0, axis=0)
                            + np.roll(self.VelocityY[:, 1:, 1], 0, axis=0)
                        )
                        / 2
                    )
                )
                / (
                    World.waterDensity
                    * self.Radius
                    * np.cos(self.LatitudesNodes)
                    * (
                        self.LongitudesNodes - np.roll(self.LongitudesNodes, -1, axis=0)
                    )[:, :, 1]
                )
            )

            velocityY[:, :, 1] += (
                # Filters
                # Between water
                (self.Volume[:, :-1, 1] > 0).astype(int)
                * (self.Volume[:, 1:, 1] > 0).astype(int)
                *
                # Between poles
                (self.AltitudesFaces > -1).astype(int)
                * (self.AltitudesFaces < 1).astype(int)
                *
                # Gradient of pressure
                0.5
                * (
                    # v_1^2
                    np.square(
                        (
                            np.roll(self.VelocityY[:, :, 1], 1, axis=1)
                            + self.VelocityY[:, :, 1]
                        )
                        / 2
                    )
                    + np.square(
                        (
                            np.pad(
                                np.roll(self.VelocityX[:, :, 1], 0, axis=0),
                                ((0, 0), (0, 1), (0, 0)),
                            )
                            + np.pad(
                                np.roll(self.VelocityX[:, :, 1], 0, axis=0),
                                ((0, 0), (0, 1), (0, 0)),
                            )
                        )
                        / 2
                    )
                    -
                    # v_2^2
                    np.square(
                        (
                            self.VelocityY[:, :, 1]
                            + np.roll(self.VelocityY[:, :, 1], -1, axis=1)
                        )
                        / 2
                    )
                    - np.square(
                        (
                            np.pad(
                                np.roll(self.VelocityX[:, :, 1], -1, axis=0),
                                ((0, 0), (0, 1), (0, 0)),
                            )
                            + np.pad(
                                np.roll(self.VelocityX[:, :, 1], -1, axis=0),
                                ((0, 0), (0, 1), (0, 0)),
                            )
                        )
                        / 2
                    )
                )
                / (
                    World.waterDensity
                    * self.Radius
                    * np.cos(self.LatitudesNodes)
                    * (
                        np.pad(self.LatitudesNodes, ((0, 0), (1, 0), (0, 0)))
                        - np.pad(self.LatitudesNodes, ((0, 0), (0, 1), (0, 0)))
                    )[:, :, 1]
                )
            )

        # Air
        temperature[:, :, 2] += (
            World.airK
            * self.AreaX[:, :, 2]
            * dt
            * (
                np.roll(self.Temperature, -1, axis=0)[:, :, 2]
                - self.Temperature[:, :, 2]
            )
            / (
                self.Radius
                * np.cos(self.LatitudesNodes)
                * (self.LongitudesNodes - np.roll(self.LongitudesNodes, -1, axis=0))[
                    :, :, 2
                ]
                * World.airCp
                * self.Mass[:, :, 2]
            )
            + World.airK
            * np.roll(self.AreaX[:, :, 2], 1, axis=0)
            * dt
            * (
                np.roll(self.Temperature, 1, axis=0)[:, :, 2]
                - self.Temperature[:, :, 2]
            )
            / (
                self.Radius
                * np.cos(self.LatitudesNodes)
                * (np.roll(self.LongitudesNodes, 1, axis=0) - self.LongitudesNodes)[
                    :, :, 2
                ]
                * World.airCp
                * self.Mass[:, :, 2]
            )
            + World.airK
            * self.AreaY[:, :-1, 2]
            * dt
            * (
                np.roll(self.Temperature, -1, axis=1)[:, :, 2]
                - self.Temperature[:, :, 2]
            )
            / (
                self.Radius
                * (np.roll(self.LatitudesNodes, 1, axis=0) - self.LatitudesNodes)[
                    :, :, 2
                ]
                * World.airCp
                * self.Mass[:, :, 2]
            )
            + World.airK
            * self.AreaY[:, 1:, 2]
            * dt
            * (
                np.roll(self.Temperature, 1, axis=1)[:, :, 2]
                - self.Temperature[:, :, 2]
            )
            / (
                self.Radius
                * (np.roll(self.LatitudesNodes, 1, axis=0) - self.LatitudesNodes)[
                    :, :, 2
                ]
                * World.airCp
                * self.Mass[:, :, 2]
            )
            + (self.Volume[:, :, 1] > 0).astype(int)
            * (
                (np.roll(self.ElevationsNodes, 1, axis=2) - self.ElevationsNodes)[
                    :, :, 2
                ]
                * World.airK
                + (
                    np.roll(self.ElevationsNodes, 1, axis=2)
                    - np.roll(self.ElevationsNodes, 0, axis=2)
                )[:, :, 0]
                * World.waterK
            )
            / (np.roll(self.ElevationsNodes, 2, axis=2) - self.ElevationsNodes)
            * self.AreaZ[:, :, 1]
            * dt
            * (
                np.roll(self.Temperature, 1, axis=2)[:, :, 2]
                - self.Temperature[:, :, 2]
            )
            / (
                self.Radius
                * (np.roll(self.ElevationsNodes, 1, axis=2) - self.ElevationsNodes)[
                    :, :, 2
                ]
                * World.airCp
                * self.Mass[:, :, 2]
            )
            + (self.Volume[:, :, 1] == 0).astype(int)
            * (
                (np.roll(self.ElevationsNodes, 1, axis=2) - self.ElevationsNodes)[
                    :, :, 2
                ]
                * World.airK
                + (
                    np.roll(self.ElevationsNodes, 2, axis=2)
                    - np.roll(self.ElevationsNodes, 1, axis=2)
                )[:, :, 0]
                * World.airK
            )
            / (np.roll(self.ElevationsNodes, 3, axis=2) - self.ElevationsNodes)
            * self.AreaZ[:, :, 2]
            * dt
            * (
                np.roll(self.Temperature, 2, axis=2)[:, :, 2]
                - self.Temperature[:, :, 2]
            )
            / (
                self.Radius
                * (np.roll(self.ElevationsNodes, 2, axis=2) - self.ElevationsNodes)[
                    :, :, 2
                ]
                * World.airCp
                * self.Mass[:, :, 2]
            )
        )
        velocityX[:, :, 2] += (
            # Gradient of pressure
            0.5
            * (
                # v_1^2
                np.square(
                    (
                        np.roll(self.VelocityX[:, :, 2], 1, axis=0)
                        + self.VelocityX[:, :, 1]
                    )
                    / 2
                )
                + np.square(
                    (
                        np.roll(self.VelocityY[:, :-1, 2], 1, axis=0)
                        + np.roll(self.VelocityY[:, 1:, 2], 1, axis=0)
                    )
                    / 2
                )
                -
                # v2^2
                np.square(
                    (
                        self.VelocityX[:, :, 2]
                        + np.roll(self.VelocityX[:, :, 2], -1, axis=0)
                    )
                    / 2
                )
                - np.square(
                    (
                        np.roll(self.VelocityY[:, :-1, 2], 0, axis=0)
                        + np.roll(self.VelocityY[:, 1:, 2], 0, axis=0)
                    )
                    / 2
                )
            )
            / (
                World.waterDensity
                * self.Radius
                * np.cos(self.LatitudesNodes)
                * (self.LongitudesNodes - np.roll(self.LongitudesNodes, -1, axis=0))[
                    :, :, 2
                ]
            )
        )

        velocityY[:, :, 2] += (
            # Filters
            # Between poles
            (self.AltitudesFaces > -1).astype(int)
            * (self.AltitudesFaces < 1).astype(int)
            *
            # Gradient of pressure
            0.5
            * (
                # v_1^2
                np.square(
                    (
                        np.roll(self.VelocityY[:, :, 2], 1, axis=1)
                        + self.VelocityY[:, :, 1]
                    )
                    / 2
                )
                + np.square(
                    (
                        np.pad(
                            np.roll(self.VelocityX[:, :, 2], 0, axis=0),
                            ((0, 0), (0, 1), (0, 0)),
                        )
                        + np.pad(
                            np.roll(self.VelocityX[:, :, 2], 0, axis=0),
                            ((0, 0), (0, 1), (0, 0)),
                        )
                    )
                    / 2
                )
                -
                # v_2^2
                np.square(
                    (
                        self.VelocityY[:, :, 1]
                        + np.roll(self.VelocityY[:, :, 2], -1, axis=1)
                    )
                    / 2
                )
                - np.square(
                    (
                        np.pad(
                            np.roll(self.VelocityX[:, :, 2], -1, axis=0),
                            ((0, 0), (0, 1), (0, 0)),
                        )
                        + np.pad(
                            np.roll(self.VelocityX[:, :, 2], -1, axis=0),
                            ((0, 0), (0, 1), (0, 0)),
                        )
                    )
                    / 2
                )
            )
            / (
                World.waterDensity
                * self.Radius
                * np.cos(self.LatitudesNodes)
                * (
                    np.pad(self.LatitudesNodes, ((0, 0), (1, 0), (0, 0)))
                    - np.pad(self.LatitudesNodes, ((0, 0), (0, 1), (0, 0)))
                )[:, :, 2]
            )
        )
        """
        dv/dt = grad(P) dA
        m/s^2 = kg*m/s^2*m^2/m
        
        d2m/dt2 = grad(P) * A
        kg/s^2 = kg*m/s^2*m^2*m * m^2
        
        d2m/dt2 = (dm/dr * R * T / g * V) + (m * R * dT/dr /g * V) * A
        
        P = m/g*R*T/V
        kg*m/s^2*m^2 = kg * mol/kg * kg*m/s^2*m/K*mol * K / m^3
        
        g = 0.029 kg/mol
        
        dm/dt / p / A = v
        """

        # Air
        # accelerationX = (np.roll(self.Mass[2], 1, axis=1) - np.roll(self.Mass[2], -1, axis=1)) / (self.Scale * self.SizeX * np.cos(J) * (np.roll(I, 1, axis=1) - np.roll(I, -1, axis=1))) * R * self.Temperature[2] / g * volume \
        #     + self.Mass[2] * R * (np.roll(self.Temperature[2], 1, axis=1) - np.roll(self.Temperature[2], -1, axis=1)) / (self.Scale * 1000 / 360 * np.cos(J) * (np.roll(I, 1, axis=1) - np.roll(I, -1, axis=1))) / g * volume * areaX

        # accelerationY = (np.concatenate([self.Mass[2][0],self.Mass[2][:-1]], axis=0) - np.concatenate([self.Mass[2][1:],self.Mass[2][-1]], axis=0)) / (self.Scale * self.SizeX * (np.concatenate([J[0],J[:-1]], axis=0) - np.concatenate([J[1:],J[-1]], axis=0))) * R * self.Temperature[2] / g * volume \
        #     + self.Mass[2] * R * (np.concatenate([self.Temperature[2][0],self.Temperature[2][:-1]], axis=0) - np.concatenate([self.Temperature[2][1:],self.Temperature[2][-1]], axis=0)) / (self.Scale * self.SizeX / 360 * (np.concatenate([J[0],J[:-1]], axis=0) - np.concatenate([J[1:],J[-1]], axis=0))) / g * volume * areaY

        self.Temperature = temperature
        self.Mass = mass
        self.VelocityX = velocityX
        self.VelocityY = velocityY


# %%

world = World(elevation, lenYear=120)
