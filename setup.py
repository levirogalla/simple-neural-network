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


def createData(iterations, repeats, shuffle=True):
    data = []
    for x in tqdm(range(int(iterations))):
        if x:
            for _ in range(repeats):
                # data.append([Vector(x), Vector(root(x))])
                data.append(h(x))
    if shuffle:
        random.shuffle(data)
    return Matrix(*data)


TRAIN = createData(ITERATIONS, REPEATS)

print("Created training data")
