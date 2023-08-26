# %%

from matplotlib import colors
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt
import itertools

# %%

elevation = np.load("world/elevation.npy").transpose()
temperature = np.moveaxis(np.load("world/temperature.npy"), 1, -1)
humidity = np.load("world/humidity.npy")
day = 0

# %%


class Node:
    X: float
    Y: float
    Z: float
    FlowRate: float

    def __init__(self, x: float, y: float, z: float, flowRate: float):
        self.X = x
        self.Y = y
        self.Z = z
        self.FlowRate = flowRate


class Edge:
    Source: Node
    Destination: Node

    def __init__(self, source: Node, destination: Node):
        self.Source = source
        self.Destination = destination


# %%

lakeEvaportation = 50
riverEvaporation = 2


def flood(x: int, y: int, elevation: np.ndarray) -> float:
    saturated = np.zeros(elevation.shape)
    edges = np.zeros(elevation.shape)
    depth = 0
    global nodes, outs

    def saturate(i: int, j: int, saturated: np.ndarray, edges: np.ndarray):
        edges[i, j] = 0
        saturated[i, j] = 1

        # North
        if j < elevation.shape[1] - 1 and saturated[i, j + 1] == 0:
            edges[i, j + 1] = 1

        # South
        if j > 0 and saturated[i, j - 1] == 0:
            edges[i, j - 1] = 1

        # East
        if saturated[i - 1, j] == 0:
            edges[i - 1, j] = 1

        # West
        if saturated[(i + 1) % elevation.shape[0], j] == 0:
            edges[(i + 1) % elevation.shape[0], j] = 1

        return saturated, edges

    outflow = 0
    outflow = sum([n.FlowRate for n in nodes[x, y]])

    while outflow >= 0:
        saturated, edges = saturate(x, y, saturated, edges)

        outflow -= lakeEvaportation
        depth = elevation[x, y]

        idxs = np.where(edges)
        x, y = np.transpose(idxs)[np.argmin(elevation[idxs])]
        if elevation[x, y] < depth:
            break

    saturation = (depth - elevation + 1e-6) * saturated

    if outflow > 0:
        # North
        if y < elevation.shape[1] - 1 and saturated[x, y + 1] == 1:
            nodes[x, y] += [Node(x + 0.5, y + 1, elevation[x, y], outflow)]

        # South
        elif y > 0 and saturated[x, y - 1] == 1:
            nodes[x, y] += [Node(x + 0.5, y, elevation[x, y], outflow)]

        # East
        elif saturated[x - 1, y] == 1:
            nodes[x, y] += [Node(x + 1, y + 0.5, elevation[x, y], outflow)]

        # West
        else:
            nodes[x, y] += [Node(x, y + 0.5, elevation[x, y], outflow)]

    return saturation


# %%

order = np.dstack(np.unravel_index(np.argsort(elevation, None)[::-1], elevation.shape))[
    0
]

# %%

nodes = np.ndarray(elevation.shape, dtype=list)
nodes[:] = [[[] for l in range(elevation.shape[1])] for l in range(elevation.shape[0])]
ins = np.ndarray(elevation.shape, dtype=object)
outs = np.ndarray(elevation.shape, dtype=object)
edges = np.ndarray(elevation.shape, dtype=list)
edges[:] = [[[] for l in range(elevation.shape[1])] for l in range(elevation.shape[0])]
saturation = np.zeros(elevation.shape)

print("starting")

n = 0
while n < len(order):
    i, j = order[n]
    if elevation[i, j] <= 0:
        break
    if saturation[i, j] > 0:
        edges[i, j] = []
        n += 1
        continue

    # if temperature[day, i, j] > 273.1 and np.min(temperature[:, i, j]) <= 273.1:
    #     nodes[i, j] += [
    #         Node(i + 0.5, j + 0.5, elevation[i, j], flowRate=humidity[i, j])
    #     ]
    if not ins[i, j]:
        ins[i, j] = Node(i + 0.5, j + 0.5, elevation[i, j], flowRate=humidity[i, j])

    localNodes = nodes[i, j] + [ins[i, j]]
    if len(localNodes) == 0:
        edges[i, j] = []
        n += 1
        continue
    if sum([n.FlowRate for n in localNodes]) < riverEvaporation:
        edges[i, j] = []
        n += 1
        continue

    sides = [elevation[i, j]]

    # North
    if j < elevation.shape[1] - 1:
        sides += [elevation[i, j + 1]]
    else:
        sides += [np.inf]

    # South
    if j > 0:
        sides += [elevation[i, j - 1]]
    else:
        sides += [np.inf]

    # East
    sides += [elevation[(i + 1) % elevation.shape[0], j]]

    # West
    sides += [elevation[i - 1, j]]

    lowest = np.argmin(sides)
    if lowest == 0:
        saturation += flood(i, j, elevation + saturation)
        order = np.dstack(
            np.unravel_index(
                np.argsort(elevation + saturation, None)[::-1], elevation.shape
            )
        )[0]
        n = (
            len(order)
            - 1
            - np.searchsorted(
                np.sort(elevation + saturation, axis=None),
                elevation[i, j] + saturation[i, j],
                "right",
            )
        )
        continue
    if outs[i, j] is None:
        if lowest == 1:
            outs[i, j] = Node(
                i + 0.5,
                j + 1,
                (elevation[i, j + 1] + elevation[i, j]) / 2,
                sum([n.FlowRate for n in localNodes]),
            )
            nodes[i, j + 1] += [outs[i, j]]
        if lowest == 2:
            outs[i, j] = Node(
                i + 0.5,
                j,
                (elevation[i, j - 1] + elevation[i, j]) / 2,
                sum([n.FlowRate for n in localNodes]),
            )
            nodes[i, j - 1] += [outs[i, j]]
        if lowest == 3:
            outs[i, j] = Node(
                i + 1,
                j + 0.5,
                (elevation[(i + 1) % elevation.shape[0], j] + elevation[i, j]) / 2,
                sum([n.FlowRate for n in localNodes]),
            )
            nodes[(i + 1) % elevation.shape[0], j] += [outs[i, j]]
        if lowest == 4:
            outs[i, j] = Node(
                i,
                j + 0.5,
                (elevation[i - 1, j] + elevation[i, j]) / 2,
                sum([n.FlowRate for n in localNodes]),
            )
            nodes[i - 1, j] += [outs[i, j]]

    outs[i, j].FlowRate = sum([n.FlowRate for n in localNodes]) - riverEvaporation

    edges[i, j] = [Edge(n, outs[i, j]) for n in localNodes]
    n += 1

# %%

x = np.linspace(0, 1, 128).reshape((-1, 1))
world1 = (1 - x) * np.array([[30.0 / 256, 33.0 / 256, 117.0 / 256, 1]]) + x * np.array(
    [[52.0 / 256, 205.0 / 256, 235.0 / 256, 1]]
)
world2 = (1 - x) * np.array([[70.0 / 256, 235.0 / 256, 52.0 / 256, 1]]) + x * np.array(
    [[235.0 / 256, 52.0 / 256, 52.0 / 256, 1]]
)
world = colors.LinearSegmentedColormap.from_list("world", np.vstack((world1, world2)))

# %%

satmask = saturation.copy()
satmask[satmask == 0] = np.nan


fig, ax = plt.subplots()
ax.pcolormesh(elevation.transpose(), cmap=world, vmin=-6.5, vmax=6.5)
ax.pcolormesh(-satmask.transpose(), cmap=world, vmin=-6.5, vmax=6.5)
ax.add_collection(
    LineCollection(
        [
            [
                [edge.Source.X, edge.Source.Y],
                [edge.Destination.X, edge.Destination.Y],
            ]
            for edge in itertools.chain.from_iterable(edges.flatten())
            if abs(edge.Source.X - edge.Destination.X) <= 1
            # and edge.Destination.FlowRate > 10
        ],
        color="blue",
        linewidth=0.1,
    )
)

# plt.xlim(600, 800)
# plt.ylim(200, 400)

# %%
