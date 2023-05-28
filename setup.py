from Python_Linear_Algebra.main import Matrix, Vector
import random
from tqdm import tqdm
from math import sin

ITERATIONS = 20
MAXAGRESSION = 1
INITWEIGHT = 1


def getRandom(weight=10):
    randomNum = random.randint(-weight, weight)

    if randomNum:
        return randomNum
    else:
        return getRandom(weight)


def f(x):
    return (3 * (x**2) - 15) + 2**x


def linear(x):
    return 2 * x


def third(x):
    return (x**3) + x**2 + 5 * x


def second(x):
    return (x**2) + x * 2 + 5


def fourth(x):
    return (x**4) + x**3 - 7 * x + 5


def fifth(x):
    return (x**5) + x**4 - 7 * x + 5


def h(x):
    return [1, 2]


print("Creating training data")


def createData(iterations):
    data = []
    for x in tqdm(range(iterations)):
        if x:
            data.append([Vector(x), Vector(second(x))])
    return Matrix(*data)


TRAIN = createData(ITERATIONS)

print("Created training data")
