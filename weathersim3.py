from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# %%

df = pd.read_csv("assets/globe2.csv")
elevation = df.to_numpy()[:, 1:]

# %%

R = 8.314  # J/K*mol
g = 0.029  # kg/mol
dt = 24 * 3600 / 1000  # s
I0 = 1.4e9  # W/km^2

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

    waterExpansion = 69 * 1e-6  # km^3/km^3*K

    sigma = 5.669e-2  # W/km^2K^4

    gravity = 9.8e-3  # km/s^2

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

        self.Temperature = self.TemperatureInit = 290 + 0 * np.cos(self.AltitudesNodes)
        self.Volume = (
            (2 * np.pi / self.SizeX)
            * (2 / self.SizeY)
            * (
                np.power(self.Radius + self.ElevationsFaces[:, :, 1:], 3)
                - np.power(self.Radius + self.ElevationsFaces[:, :, :-1], 3)
            )
            / 3
        )
        self.Mass = np.copy(self.Volume)
        self.Mass[:, :, 0] *= World.earthDensity
        self.Mass[:, :, 1] *= World.waterDensity
        self.Mass[:, :, 2] *= World.airDensity
        with np.errstate(divide="ignore", invalid="ignore"):
            self.VelocityX = self.VelocityXInitX = np.nan_to_num(
                2
                * np.pi
                / (self.LenDay * 3600)
                # 1/(r2-r1)Swrdr = w/(r2-r1)*1/2*(r2^2-r1^2)
                * (
                    0.5
                    * (
                        self.ElevationsXFaces[:, :, 1:]
                        - self.ElevationsXFaces[:, :, :-1]
                    )
                    + self.Radius
                )
                * np.cos(self.AltitudesNodes)
            )

            self.VelocityXInitY = np.nan_to_num(
                2
                * np.pi
                / (self.LenDay * 3600)
                # 1/(r2-r1)Swrdr = w/(r2-r1)*1/2*(r2^2-r1^2)
                * (
                    0.5
                    * (
                        self.ElevationsYFaces[:, :, 1:]
                        - self.ElevationsYFaces[:, :, :-1]
                    )
                    + self.Radius
                )
                * np.cos(self.AltitudesFaces)
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
        # Next iteration is initially copy
        temperature = np.copy(self.Temperature)
        mass = np.copy(self.Mass)
        velocityX = np.copy(self.VelocityX)
        velocityY = np.copy(self.VelocityY)

        # region Earth
        # Update temperature
        temperature[:, :, 0] += (
            (  # East
                World.earthK
                * self.AreaX[:, :, 0]
                * dt
                * (
                    np.roll(self.Temperature, -1, axis=0)[:, :, 0]
                    - self.Temperature[:, :, 0]
                )
                / (
                    self.Radius
                    * np.cos(self.AltitudesNodes)[:, :, 0]
                    * (self.AzimuthsNodes - np.roll(self.AzimuthsNodes, -1, axis=0))[
                        :, :, 0
                    ]
                    * World.earthCp
                    * self.Mass[:, :, 0]
                )
            )
            + (  # West
                World.earthK
                * np.roll(self.AreaX[:, :, 0], 1, axis=0)
                * dt
                * (
                    np.roll(self.Temperature, 1, axis=0)[:, :, 0]
                    - self.Temperature[:, :, 0]
                )
                / (
                    self.Radius
                    * np.cos(self.AltitudesNodes)[:, :, 0]
                    * (np.roll(self.AzimuthsNodes, 1, axis=0) - self.AzimuthsNodes)[
                        :, :, 0
                    ]
                    * World.earthCp
                    * self.Mass[:, :, 0]
                )
            )
            + (  # North
                World.earthK
                * self.AreaY[:, :-1, 0]
                * dt
                * (
                    np.pad(self.Temperature[:, 1:, 0], ((0, 0), (0, 1)))
                    - np.pad(self.Temperature[:, 1:, 0], ((0, 0), (0, 1)))
                )
                / (
                    self.Radius
                    * (np.roll(self.AltitudesNodes, -1, axis=1) - self.AltitudesNodes)[
                        :, :, 0
                    ]
                    * World.earthCp
                    * self.Mass[:, :, 0]
                )
            )
            + (  # South
                World.earthK
                * self.AreaY[:, 1:, 0]
                * dt
                * (
                    np.pad(self.Temperature[:, :-1, 0], ((0, 0), (1, 0)))
                    - np.pad(self.Temperature[:, 1:, 0], ((0, 0), (1, 0)))
                )
                / (
                    self.Radius
                    * (np.roll(self.AltitudesNodes, 1, axis=1) - self.AltitudesNodes)[
                        :, :, 0
                    ]
                    * World.earthCp
                    * self.Mass[:, :, 0]
                )
            )
            + (  # Water
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
                / (np.roll(self.ElevationsNodes, 1, axis=2) - self.ElevationsNodes)[
                    :, :, 0
                ]
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
            + (  # Air
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
                / (np.roll(self.ElevationsNodes, 2, axis=2) - self.ElevationsNodes)[
                    :, :, 0
                ]
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
            + (  # Sunlight
                (self.Volume[:, :, 1] == 0).astype(int)
                * self.AreaZ[:, :, 1]
                * dt
                / (self.Mass[:, :, 0] * self.earthCp)
                * intensity
            )
            - (  # Radiation
                (self.Volume[:, :, 1] == 0).astype(int)
                * self.AreaZ[:, :, 1]
                * dt
                * World.sigma
                * np.power(self.Temperature[:, :, 0], 4)
                / (self.Mass[:, :, 0] * self.earthCp)
            )
        )
        # endregion

        # region Water
        with np.errstate(divide="ignore", invalid="ignore"):
            # motion
            temperature[:, :, 1] = np.nan_to_num(
                self.Mass[:, :, 1]
                * self.Temperature[:, :, 1]
                * (
                    (self.Volume[:, :, 1] > 0).astype(int)
                    * dt
                    / World.waterDensity
                    * (
                        # East
                        -(
                            np.roll(
                                self.Temperature[:, :, 1] * self.Volume[:, :, 1],
                                -1,
                                axis=1,
                            )
                            > 0
                        ).astype(int)
                        * np.roll(
                            (self.VelocityX - self.VelocityXInitX)[:, :, 1]
                            * self.AreaX[:, :, 1],
                            -1,
                            axis=1,
                        )
                        # West
                        + (
                            np.roll(
                                self.Temperature[:, :, 1] * self.Volume[:, :, 1],
                                1,
                                axis=1,
                            )
                            > 0
                        ).astype(int)
                        * (self.VelocityX - self.VelocityXInitX)[:, :, 1]
                        * self.AreaX[:, :, 1]
                        # North
                        - (
                            np.pad(
                                self.Temperature[:, :, 1] * self.Volume[:, :, 1],
                                ((0, 0), (0, 1)),
                            )
                            > 0
                        ).astype(int)[:, 1:]
                        * self.VelocityY[:, 1:, 1]
                        * self.AreaY[:, 1:, 1]
                        # South
                        + (
                            np.pad(
                                self.Temperature[:, :, 1] * self.Volume[:, :, 1],
                                ((0, 0), (1, 0)),
                            )
                            > 0
                        ).astype(int)[:, :-1]
                        * self.VelocityY[:, :-1, 1]
                        * self.AreaY[:, :-1, 1]
                    )
                )
                / (
                    self.Mass[:, :, 1]
                    + (self.Volume[:, :, 1] > 0).astype(int)
                    * dt
                    / World.waterDensity
                    * (
                        # East
                        -(np.roll(self.Volume[:, :, 1], -1, axis=1) > 0).astype(int)
                        * np.roll(
                            (self.VelocityX - self.VelocityXInitX)[:, :, 1]
                            * self.AreaX[:, :, 1],
                            -1,
                            axis=1,
                        )
                        # West
                        + (np.roll(self.Volume[:, :, 1], 1, axis=1) > 0).astype(int)
                        * (self.VelocityX - self.VelocityXInitX)[:, :, 1]
                        * self.AreaX[:, :, 1]
                        # North
                        - (np.pad(self.Volume[:, :, 1], ((0, 0), (0, 1))) > 0).astype(
                            int
                        )[:, 1:]
                        * self.VelocityY[:, 1:, 1]
                        * self.AreaY[:, 1:, 1]
                        # South
                        + (np.pad(self.Volume[:, :, 1], ((0, 0), (1, 0))) > 0).astype(
                            int
                        )[:, :-1]
                        * self.VelocityY[:, :-1, 1]
                        * self.AreaY[:, :-1, 1]
                    )
                )
            )

            # conduction
            temperature[:, :, 1] += np.nan_to_num(
                (  # East
                    World.waterK
                    * self.AreaX[:, :, 1]
                    * dt
                    * (
                        np.roll(self.Temperature, -1, axis=0)[:, :, 1]
                        - self.Temperature[:, :, 1]
                    )
                    / (
                        self.Radius
                        * np.cos(self.AltitudesNodes[:, :, 1])
                        * (
                            self.AzimuthsNodes - np.roll(self.AzimuthsNodes, -1, axis=0)
                        )[:, :, 1]
                        * World.waterCp
                        * self.Mass[:, :, 1]
                    )
                )
                + (  # West
                    World.waterK
                    * np.roll(self.AreaX[:, :, 1], 1, axis=0)
                    * dt
                    * (
                        np.roll(self.Temperature, 1, axis=0)[:, :, 1]
                        - self.Temperature[:, :, 1]
                    )
                    / (
                        self.Radius
                        * np.cos(self.AltitudesNodes[:, :, 1])
                        * (np.roll(self.AzimuthsNodes, 1, axis=0) - self.AzimuthsNodes)[
                            :, :, 1
                        ]
                        * World.waterCp
                        * self.Mass[:, :, 1]
                    )
                )
                + (  # North
                    World.waterK
                    * self.AreaY[:, :-1, 1]
                    * dt
                    * (
                        np.pad(self.Temperature[:, 1:, 1], ((0, 0), (0, 1)))
                        - np.pad(self.Temperature[:, 1:, 1], ((0, 0), (0, 1)))
                    )
                    / (
                        self.Radius
                        * (
                            np.roll(self.AltitudesNodes, -1, axis=1)
                            - self.AltitudesNodes
                        )[:, :, 1]
                        * World.waterCp
                        * self.Mass[:, :, 1]
                    )
                )
                + (  # South
                    World.waterK
                    * self.AreaY[:, 1:, 1]
                    * dt
                    * (
                        np.pad(self.Temperature[:, :-1, 1], ((0, 0), (1, 0)))
                        - np.pad(self.Temperature[:, 1:, 1], ((0, 0), (1, 0)))
                    )
                    / (
                        self.Radius
                        * (
                            np.roll(self.AltitudesNodes, 1, axis=1)
                            - self.AltitudesNodes
                        )[:, :, 1]
                        * World.waterCp
                        * self.Mass[:, :, 1]
                    )
                )
                + (
                    (  # Earth
                        (
                            np.roll(self.ElevationsNodes, 1, axis=2)
                            - self.ElevationsNodes
                        )[:, :, 0]
                        * World.earthK
                        + (
                            np.roll(self.ElevationsNodes, 2, axis=2)
                            - np.roll(self.ElevationsNodes, 1, axis=2)
                        )[:, :, 0]
                        * World.waterK
                    )
                    / (np.roll(self.ElevationsNodes, 1, axis=2) - self.ElevationsNodes)[
                        :, :, 0
                    ]
                    * self.AreaZ[:, :, 1]
                    * dt
                    * (
                        np.roll(self.Temperature, 1, axis=2)[:, :, 0]
                        - self.Temperature[:, :, 0]
                    )
                    / (
                        self.Radius
                        * (
                            np.roll(self.ElevationsNodes, 1, axis=2)
                            - self.ElevationsNodes
                        )[:, :, 0]
                        * World.waterCp
                        * self.Mass[:, :, 0]
                    )
                )
                + (  # Air
                    (
                        (
                            np.roll(self.ElevationsNodes, 2, axis=2)
                            - np.roll(self.ElevationsNodes, 1, axis=2)
                        )[:, :, 0]
                        * World.waterK
                        + (
                            np.roll(self.ElevationsNodes, 3, axis=2)
                            - np.roll(self.ElevationsNodes, 2, axis=2)
                        )[:, :, 0]
                        * World.airK
                    )
                    / (
                        np.roll(self.ElevationsNodes, 3, axis=2)
                        - np.roll(self.ElevationsNodes, 2, axis=2)
                    )[:, :, 2]
                    * self.AreaZ[:, :, 2]
                    * dt
                    * (
                        np.roll(self.Temperature, 2, axis=2)[:, :, 1]
                        - self.Temperature[:, :, 1]
                    )
                    / (
                        self.Radius
                        * (
                            np.roll(self.ElevationsNodes, 2, axis=2)
                            - self.ElevationsNodes
                        )[:, :, 1]
                        * World.waterCp
                        * self.Mass[:, :, 1]
                    )
                )
                + (  # Sunlight
                    self.AreaZ[:, :, 2]
                    * dt
                    / (self.Mass[:, :, 1] * self.earthCp)
                    * intensity
                )
                - (  # Radiation
                    self.AreaZ[:, :, 2]
                    * dt
                    * World.sigma
                    * np.power(self.Temperature[:, :, 1], 4)
                    / (self.Mass[:, :, 1] * self.waterCp)
                )
            )

            mass[:, :, 1] += np.nan_to_num(
                (self.Volume[:, :, 1] > 0).astype(int)
                * dt
                / World.waterDensity
                * (
                    # East
                    -(np.roll(self.Volume[:, :, 1], -1, axis=1) > 0).astype(int)
                    * np.roll(
                        (self.VelocityX - self.VelocityXInitX)[:, :, 1]
                        * self.AreaX[:, :, 1],
                        -1,
                        axis=1,
                    )
                    # West
                    + (np.roll(self.Volume[:, :, 1], 1, axis=1) > 0).astype(int)
                    * (self.VelocityX - self.VelocityXInitX)[:, :, 1]
                    * self.AreaX[:, :, 1]
                    # North
                    - (np.pad(self.Volume[:, :, 1], ((0, 0), (0, 1))) > 0).astype(int)[
                        :, 1:
                    ]
                    * self.VelocityY[:, 1:, 1]
                    * self.AreaY[:, 1:, 1]
                    # South
                    + (np.pad(self.Volume[:, :, 1], ((0, 0), (1, 0))) > 0).astype(int)[
                        :, :-1
                    ]
                    * self.VelocityY[:, :-1, 1]
                    * self.AreaY[:, :-1, 1]
                )
            )

            # v_x' = dP/dx / p
            # dP/dx = (P_x+1 - P_x-1) / dx = 1/2p((v_x-1)^2 - (v_x+1)^2)
            #

            velocityX[:, :, 1] += np.nan_to_num(
                # Filters
                # Between water
                (self.Volume[:, :, 1] > 0).astype(int)
                * (np.roll(self.Volume[:, :, 1], 1, axis=1) > 0).astype(int)
                # Time integral
                * dt
                # Gradient of pressure
                * -(
                    # P_1
                    np.roll(self.Mass[:, :, 1] / self.Volume[:, :, 1], 1, axis=0)
                    / np.roll(
                        np.power(
                            1
                            + World.waterExpansion
                            * (
                                self.Temperature[:, :, 1]
                                - self.TemperatureInit[:, :, 1]
                            ),
                            3,
                        ),
                        1,
                        axis=0,
                    )
                    * (
                        # z_1
                        World.gravity * self.ElevationsXFaces[:, :, 1]
                        # v_1^2
                        + 0.5
                        * (
                            np.square(
                                (
                                    np.roll(
                                        self.VelocityX[:, :, 1]
                                        - self.VelocityXInitX[:, :, 1],
                                        1,
                                        axis=0,
                                    )
                                    + self.VelocityX[:, :, 1]
                                    - self.VelocityXInitX[:, :, 1]
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
                        )
                    )
                    # P_2
                    - self.Mass[:, :, 1]
                    / self.Volume[:, :, 1]
                    / np.roll(
                        np.power(
                            1
                            + World.waterExpansion
                            * (
                                self.Temperature[:, :, 1]
                                - self.TemperatureInit[:, :, 1]
                            ),
                            3,
                        ),
                        0,
                        axis=0,
                    )
                    * (
                        # z_2
                        World.gravity * self.ElevationsXFaces[:, :, 1]
                        # v2^2
                        + 0.5
                        * (
                            np.square(
                                (
                                    self.VelocityX[:, :, 1]
                                    - self.VelocityXInitX[:, :, 1]
                                    + np.roll(
                                        self.VelocityX[:, :, 1]
                                        - self.VelocityXInitX[:, :, 1],
                                        -1,
                                        axis=0,
                                    )
                                )
                                / 2
                            )
                            + np.square(
                                (
                                    np.roll(self.VelocityY[:, :-1, 1], 0, axis=0)
                                    + np.roll(self.VelocityY[:, 1:, 1], 0, axis=0)
                                )
                                / 2
                            )
                        )
                    )
                )
                / (
                    0.5
                    * (
                        np.roll(
                            self.Mass[:, :, 1]
                            / self.Volume[:, :, 1]
                            / np.power(
                                1
                                + World.waterExpansion
                                * (
                                    self.Temperature[:, :, 1]
                                    - self.TemperatureInit[:, :, 1]
                                ),
                                3,
                            ),
                            1,
                            axis=0,
                        )
                        + self.Mass[:, :, 1]
                        / self.Volume[:, :, 1]
                        / np.power(
                            1
                            + World.waterExpansion
                            * (
                                self.Temperature[:, :, 1]
                                - self.TemperatureInit[:, :, 1]
                            ),
                            3,
                        )
                    )
                    * self.Radius
                    * np.cos(self.AltitudesNodes[:, :, 1])
                    * (self.AzimuthsNodes - np.roll(self.AzimuthsNodes, 1, axis=0))[
                        :, :, 1
                    ]
                )
            )

            velocityY[:, :, 1] += np.nan_to_num(
                # Filters
                # Between water
                (np.pad(self.Volume[:, :, 1], ((0, 0), (1, 0))) > 0).astype(int)
                * (np.pad(self.Volume[:, :, 1], ((0, 0), (0, 1))) > 0).astype(int)
                # Between poles
                * (self.AltitudesFaces[:, :, 1] > -1).astype(int)
                * (self.AltitudesFaces[:, :, 1] < 1).astype(int)
                # Time integral
                * dt
                # Gradient of pressure
                * -(
                    # P_1
                    np.pad(
                        self.Mass[:, :, 1] / self.Volume[:, :, 1],
                        ((0, 0), (0, 1)),
                    )
                    / np.pad(
                        np.power(
                            1
                            + World.waterExpansion
                            * (
                                self.Temperature[:, :, 1]
                                - self.TemperatureInit[:, :, 1]
                            ),
                            3,
                        ),
                        ((0, 0), (0, 1)),
                    )
                    * (
                        # z_1
                        World.gravity * self.ElevationsYFaces[:, :, 1]
                        # v_1^2
                        + 0.5
                        * (
                            np.square(
                                (
                                    np.roll(self.VelocityY[:, :, 1], 1, axis=1)
                                    + self.VelocityY[:, :, 1]
                                )
                                / 2
                            )
                            + np.roll(
                                np.square(
                                    (
                                        np.pad(
                                            np.roll(
                                                self.VelocityX[:, :, 1]
                                                - self.VelocityXInitX[:, :, 1],
                                                -1,
                                                axis=0,
                                            ),
                                            ((0, 0), (0, 1)),
                                        )
                                        + np.pad(
                                            np.roll(
                                                self.VelocityX[:, :, 1]
                                                - self.VelocityXInitX[:, :, 1],
                                                0,
                                                axis=0,
                                            ),
                                            ((0, 0), (0, 1)),
                                        )
                                    )
                                    / 2
                                ),
                                1,
                                axis=1,
                            )
                        )
                    )
                    # P_2
                    - np.pad(
                        self.Mass[:, :, 1] / self.Volume[:, :, 1],
                        ((0, 0), (1, 0)),
                    )
                    / np.pad(
                        np.power(
                            1
                            + World.waterExpansion
                            * (
                                self.Temperature[:, :, 1]
                                - self.TemperatureInit[:, :, 1]
                            ),
                            3,
                        ),
                        ((0, 0), (1, 0)),
                    )
                    * (
                        # z_2
                        World.gravity * self.ElevationsYFaces[:, :, 1]
                        # v_2^2
                        + 0.5
                        * (
                            np.square(
                                (
                                    self.VelocityY[:, :, 1]
                                    + np.roll(self.VelocityY[:, :, 1], -1, axis=1)
                                )
                                / 2
                            )
                            + np.square(
                                (
                                    np.pad(
                                        np.roll(self.VelocityX[:, :, 1], -1, axis=0),
                                        ((0, 0), (0, 1)),
                                    )
                                    + np.pad(
                                        np.roll(self.VelocityX[:, :, 1], 0, axis=0),
                                        ((0, 0), (0, 1)),
                                    )
                                )
                                / 2
                                - self.VelocityXInitY[:, :, 1]
                            )
                        )
                    )
                )
                / (
                    0.5
                    * (
                        np.pad(
                            self.Mass[:, :, 1]
                            / self.Volume[:, :, 1]
                            / np.power(
                                1
                                + World.waterExpansion
                                * (
                                    self.Temperature[:, :, 1]
                                    - self.TemperatureInit[:, :, 1]
                                ),
                                3,
                            ),
                            ((0, 0), (0, 1)),
                        )
                        + np.pad(
                            self.Mass[:, :, 1]
                            / self.Volume[:, :, 1]
                            / np.power(
                                1
                                + World.waterExpansion
                                * (
                                    self.Temperature[:, :, 1]
                                    - self.TemperatureInit[:, :, 1]
                                ),
                                3,
                            ),
                            ((0, 0), (1, 0)),
                        )
                    )
                    * self.Radius
                    * (
                        np.pad(self.AltitudesNodes, ((0, 0), (0, 1), (0, 0)))
                        - np.pad(self.AltitudesNodes, ((0, 0), (1, 0), (0, 0)))
                    )[:, :, 1]
                )
            )
        # endregion

        # region Air
        # motion
        temperature[:, :, 2] = (
            self.Mass[:, :, 2]
            * self.Temperature[:, :, 2]
            * (
                dt
                / World.airDensity
                * (
                    # East
                    -np.roll(
                        (self.VelocityX - self.VelocityXInitX)[:, :, 2]
                        * self.AreaX[:, :, 2],
                        -1,
                        axis=1,
                    )
                    # West
                    + (self.VelocityX - self.VelocityXInitX)[:, :, 2]
                    * self.AreaX[:, :, 2]
                    # North
                    - self.VelocityY[:, 1:, 2] * self.AreaY[:, 1:, 2]
                    # South
                    + self.VelocityY[:, :-1, 2] * self.AreaY[:, :-1, 2]
                )
            )
            / (
                self.Mass[:, :, 2]
                + dt
                / World.airDensity
                * (
                    # East
                    -np.roll(
                        (self.VelocityX - self.VelocityXInitX)[:, :, 2]
                        * self.AreaX[:, :, 2],
                        -1,
                        axis=1,
                    )
                    # West
                    + (self.VelocityX - self.VelocityXInitX)[:, :, 2]
                    * self.AreaX[:, :, 2]
                    # North
                    - self.VelocityY[:, 1:, 2] * self.AreaY[:, 1:, 2]
                    # South
                    + self.VelocityY[:, :-1, 2] * self.AreaY[:, :-1, 2]
                )
            )
        )

        # conduction
        temperature[:, :, 2] += (
            (  # East
                World.airK
                * self.AreaX[:, :, 2]
                * dt
                * (
                    np.roll(self.Temperature, -1, axis=0)[:, :, 2]
                    - self.Temperature[:, :, 2]
                )
                / (
                    self.Radius
                    * np.cos(self.AltitudesNodes[:, :, 2])
                    * (self.AzimuthsNodes - np.roll(self.AzimuthsNodes, -1, axis=0))[
                        :, :, 2
                    ]
                    * World.airCp
                    * self.Mass[:, :, 2]
                )
            )
            + (  # West
                World.airK
                * np.roll(self.AreaX[:, :, 2], 1, axis=0)
                * dt
                * (
                    np.roll(self.Temperature, 1, axis=0)[:, :, 2]
                    - self.Temperature[:, :, 2]
                )
                / (
                    self.Radius
                    * np.cos(self.AltitudesNodes[:, :, 2])
                    * (np.roll(self.AzimuthsNodes, 1, axis=0) - self.AzimuthsNodes)[
                        :, :, 2
                    ]
                    * World.airCp
                    * self.Mass[:, :, 2]
                )
            )
            + (  # North
                World.airK
                * self.AreaY[:, :-1, 2]
                * dt
                * (
                    np.pad(self.Temperature[:, 1:, 2], ((0, 0), (0, 1)))
                    - np.pad(self.Temperature[:, 1:, 2], ((0, 0), (0, 1)))
                )
                / (
                    self.Radius
                    * (np.roll(self.AltitudesNodes, -1, axis=1) - self.AltitudesNodes)[
                        :, :, 2
                    ]
                    * World.airCp
                    * self.Mass[:, :, 2]
                )
            )
            + (  # South
                World.airK
                * self.AreaY[:, 1:, 2]
                * dt
                * (
                    np.pad(self.Temperature[:, :-1, 2], ((0, 0), (1, 0)))
                    - np.pad(self.Temperature[:, 1:, 2], ((0, 0), (1, 0)))
                )
                / (
                    self.Radius
                    * (np.roll(self.AltitudesNodes, 1, axis=1) - self.AltitudesNodes)[
                        :, :, 2
                    ]
                    * World.airCp
                    * self.Mass[:, :, 2]
                )
            )
            + (  # Water
                (self.Volume[:, :, 1] > 0).astype(int)
                * (
                    (np.roll(self.ElevationsNodes, 1, axis=2) - self.ElevationsNodes)[
                        :, :, 2
                    ]
                    * World.airK
                    + (
                        np.roll(self.ElevationsNodes, 1, axis=2)
                        - np.roll(self.ElevationsNodes, 0, axis=2)
                    )[:, :, 2]
                    * World.waterK
                )
                / (np.roll(self.ElevationsNodes, 2, axis=2) - self.ElevationsNodes)[
                    :, :, 1
                ]
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
            )
            + (  # Earth
                (self.Volume[:, :, 1] == 0).astype(int)
                * (
                    (np.roll(self.ElevationsNodes, 1, axis=2) - self.ElevationsNodes)[
                        :, :, 2
                    ]
                    * World.airK
                    + (
                        np.roll(self.ElevationsNodes, 2, axis=2)
                        - np.roll(self.ElevationsNodes, 1, axis=2)
                    )[:, :, 0]
                    * World.earthK
                )
                / (np.roll(self.ElevationsNodes, 1, axis=2) - self.ElevationsNodes)[
                    :, :, 2
                ]
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
        )

        mass[:, :, 2] += (
            dt
            / World.airDensity
            * (
                # East
                -np.roll(
                    (self.VelocityX - self.VelocityXInitX)[:, :, 2]
                    * self.AreaX[:, :, 2],
                    -1,
                    axis=1,
                )
                # West
                + (self.VelocityX - self.VelocityXInitX)[:, :, 2] * self.AreaX[:, :, 2]
                # North
                - self.VelocityY[:, 1:, 2] * self.AreaY[:, 1:, 2]
                # South
                + self.VelocityY[:, :-1, 2] * self.AreaY[:, :-1, 2]
            )
        )

        velocityX[:, :, 2] += (
            # Time integral
            dt
            # Gradient of pressure
            * -(
                # P_1
                np.roll(self.Mass[:, :, 2] / self.Volume[:, :, 2], 1, axis=0)
                * (
                    # z_1
                    World.gravity * self.ElevationsXFaces[:, :, 2]
                    # v_1^2
                    + 0.5
                    * (
                        np.square(
                            (
                                np.roll(
                                    self.VelocityX[:, :, 2]
                                    - self.VelocityXInitX[:, :, 2],
                                    1,
                                    axis=0,
                                )
                                + self.VelocityX[:, :, 2]
                                - self.VelocityXInitX[:, :, 2]
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
                    )
                )
                # P_2
                - self.Mass[:, :, 2]
                / self.Volume[:, :, 2]
                * (
                    # z_2
                    World.gravity * self.ElevationsXFaces[:, :, 2]
                    # v2^2
                    + 0.5
                    * (
                        np.square(
                            (
                                self.VelocityX[:, :, 2]
                                - self.VelocityXInitX[:, :, 2]
                                + np.roll(
                                    self.VelocityX[:, :, 2]
                                    - self.VelocityXInitX[:, :, 2],
                                    -1,
                                    axis=0,
                                )
                            )
                            / 2
                        )
                        + np.square(
                            (
                                np.roll(self.VelocityY[:, :-1, 2], 0, axis=0)
                                + np.roll(self.VelocityY[:, 1:, 2], 0, axis=0)
                            )
                            / 2
                        )
                    )
                )
            )
            / (
                0.5
                * (
                    np.roll(self.Mass[:, :, 2] / self.Volume[:, :, 2], 1, axis=0)
                    + self.Mass[:, :, 2] / self.Volume[:, :, 2]
                )
                * self.Radius
                * np.cos(self.AltitudesNodes[:, :, 2])
                * (self.AzimuthsNodes - np.roll(self.AzimuthsNodes, 1, axis=0))[:, :, 2]
            )
        )

        velocityY[:, :, 2] += (
            # Filters
            # Between poles
            (self.AltitudesFaces[:, :, 2] > -1).astype(int)
            * (self.AltitudesFaces[:, :, 2] < 1).astype(int)
            # Time integral
            * dt
            # Gradient of pressure
            * -(
                # P_1
                np.pad(
                    self.Mass[:, :, 2] / self.Volume[:, :, 2],
                    ((0, 0), (0, 1)),
                )
                * (
                    # z_1
                    World.gravity * self.ElevationsYFaces[:, :, 2]
                    # v_1^2
                    + 0.5
                    * (
                        np.square(
                            (
                                np.roll(self.VelocityY[:, :, 2], 1, axis=1)
                                + self.VelocityY[:, :, 2]
                            )
                            / 2
                        )
                        + np.roll(
                            np.square(
                                (
                                    np.pad(
                                        np.roll(
                                            self.VelocityX[:, :, 2]
                                            - self.VelocityXInitX[:, :, 2],
                                            -1,
                                            axis=0,
                                        ),
                                        ((0, 0), (0, 1)),
                                    )
                                    + np.pad(
                                        np.roll(
                                            self.VelocityX[:, :, 2]
                                            - self.VelocityXInitX[:, :, 2],
                                            0,
                                            axis=0,
                                        ),
                                        ((0, 0), (0, 1)),
                                    )
                                )
                                / 2
                            ),
                            1,
                            axis=1,
                        )
                    )
                )
                - np.pad(
                    self.Mass[:, :, 2] / self.Volume[:, :, 2],
                    ((0, 0), (1, 0)),
                )
                * (
                    # z_2
                    World.gravity * self.ElevationsYFaces[:, :, 2]
                    # v_2^2
                    + 0.5
                    * (
                        np.square(
                            (
                                self.VelocityY[:, :, 2]
                                + np.roll(self.VelocityY[:, :, 2], -1, axis=1)
                            )
                            / 2
                        )
                        + np.square(
                            (
                                np.pad(
                                    np.roll(
                                        self.VelocityX[:, :, 2]
                                        - self.VelocityXInitX[:, :, 2],
                                        -1,
                                        axis=0,
                                    ),
                                    ((0, 0), (0, 1)),
                                )
                                + np.pad(
                                    np.roll(
                                        self.VelocityX[:, :, 2]
                                        - self.VelocityXInitX[:, :, 2],
                                        0,
                                        axis=0,
                                    ),
                                    ((0, 0), (0, 1)),
                                )
                            )
                            / 2
                        )
                    )
                )
            )
            / (
                0.5
                * (
                    np.pad(
                        self.Mass[:, :, 2] / self.Volume[:, :, 2],
                        ((0, 0), (0, 1)),
                    )
                    + np.pad(
                        self.Mass[:, :, 2] / self.Volume[:, :, 2],
                        ((0, 0), (1, 0)),
                    )
                )
                * self.Radius
                * (
                    np.pad(self.AltitudesNodes, ((0, 0), (0, 1), (0, 0)))
                    - np.pad(self.AltitudesNodes, ((0, 0), (1, 0), (0, 0)))
                )[:, :, 2]
            )
        )
        # endregion
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
x1, y1, t1 = world.VelocityX, world.VelocityY, world.Temperature
for i in range(10):
    world.Update(I0)
    print(
        np.nan_to_num((world.Mass / world.Volume)[:, :, 1], nan=1e12).max(),
        (world.VelocityX - x1)[:, :, 1].max(),
        world.VelocityY[:, :, 1].max(),
    )
plt.figure()
# plt.pcolor((world.Temperature - t1)[:, :, 1].transpose())
plt.streamplot(
    np.arange(100),
    np.arange(100),
    (world.VelocityX[100:200, 100:200, 1] - x1[100:200, 100:200, 1]),
    world.VelocityY[100:200, 100:200, 1] - y1[100:200, 100:200, 1],
    # scale=max(
    #     np.max(
    #         np.abs(
    #             world.VelocityX[::50, ::50, 2].transpose()
    #             - x1[::50, ::50, 2].transpose()
    #         )
    #     ),
    #     np.max(
    #         np.abs(
    #             world.VelocityY[::50, 1::50, 2].transpose()
    #             - y1[::50, 1::50, 2].transpose()
    #         )
    #     ),
    # ),
    # angles="xy",
    # scale_units="xy",
)

plt.pcolor((world.VelocityX[100:200, 100:200, 1] - x1[100:200, 100:200, 1]))
