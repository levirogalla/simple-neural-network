from Python_Linear_Algebra.main import Matrix, Vector
import random
from tqdm import tqdm
from math import sin

ITERATIONS = 1000
MAXAGRESSION = 1
INITWEIGHT = 1
REPEATS = 1
TestTRAIN = Matrix(
    [Vector(1), Vector(1)],
    [Vector(2), Vector(4)],
    [Vector(1), Vector(1)],
    [Vector(3), Vector(9)],
    [Vector(1), Vector(1)],
    [Vector(2), Vector(4)],
    [Vector(1), Vector(1)],
    [Vector(1), Vector(1)],
    [Vector(3), Vector(9)],
    [Vector(3), Vector(9)],
    [Vector(1), Vector(1)],
    [Vector(1), Vector(1)],
    [Vector(1), Vector(1)],
    [Vector(2), Vector(4)],
    [Vector(1), Vector(1)],
    [Vector(3), Vector(9)],
    [Vector(1), Vector(1)],
    [Vector(2), Vector(4)],
    [Vector(1), Vector(1)],
    [Vector(1), Vector(1)],
    [Vector(3), Vector(9)],
    [Vector(3), Vector(9)],
    [Vector(1), Vector(1)],
    [Vector(1), Vector(1)],
    [Vector(1), Vector(1)],
    [Vector(2), Vector(4)],
    [Vector(1), Vector(1)],
    [Vector(3), Vector(9)],
    [Vector(1), Vector(1)],
    [Vector(2), Vector(4)],
    [Vector(1), Vector(1)],
    [Vector(1), Vector(1)],
    [Vector(3), Vector(9)],
    [Vector(3), Vector(9)],
    [Vector(1), Vector(1)],
    [Vector(1), Vector(1)],
)


def linear(x):
    return 2 * x


def third(x):
    return x**3


def second(x):
    return x**2


def fourth(x):
    return (x**4) + x**3 - 7 * x + 5


def fifth(x):
    return (x**5) + x**4 - 7 * x + 5


def h(x):
    return [Vector(1), Vector(2)]


def root(x):
    return (abs(x)) ** 0.5


print("Creating training data")


def createData(
    dataRange: tuple[int, int],
    dataPoints: int,
    reapets: int = 0,
    shuffleData=True,
    normaliseFactor: int = 1,
) -> Vector:
    "Function creates points evenly spaced based on inputs"

    # make inputs for the funtion
    data = []
    for _ in range(reapets):
        for num in range(dataPoints):
            dataInput = num

            # squishes all numbers in input to be able to fit into given range
            dataRangeFactor = dataPoints / (dataRange[1] - dataRange[0])
            dataInput = dataInput / dataRangeFactor

            # moves point so it is relative to starting point
            dataInput = dataInput + dataRange[0]

            # normalizes
            dataInput = dataInput * (normaliseFactor)

            # applies funtion to input to get out put
            dataOutput = third(dataInput)

            data.append([Vector(dataInput), Vector(dataOutput)])

    # shuffels data
    if shuffleData:
        random.shuffle(data)
    return data


TRAIN = createData((0, 5), 5, 3, shuffleData=False)

# def createData(iterations, repeats, shuffle=True):
#     data = []
#     for x in tqdm(range(int(iterations))):
#         if x:
#             for _ in range(repeats):
#                 # data.append([Vector(x), Vector(root(x))])
#                 data.append(h(x))
#     if shuffle:
#         random.shuffle(data)
#     return Matrix(*data)


# TRAIN = createData(ITERATIONS, REPEATS)

print("Created training data")
