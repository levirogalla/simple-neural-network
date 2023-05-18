from Python_Linear_Algebra.main import Matrix, Vector
import random
from tqdm import tqdm

def getRandom(): 
    randomNum = random.randint(-10,10) 
    if randomNum:
        return randomNum
    else:
        return getRandom()

ITERATIONS = 100000
MAXAGRESSION = 1
INITWEIGHT = 1
 
# one node
layer1 = Vector()
weights12 = Matrix([getRandom()], [getRandom()], [getRandom()], [getRandom()], [getRandom()], [getRandom()])
# two nodes
layer2 = Vector()
weights23 = Matrix([getRandom(), getRandom(), getRandom(), getRandom(), getRandom(), getRandom()])
# one node
layer3 = Vector()


def f(x): 
    return ((3*(x**2) - 15) + 2**x)
def g(x):
    return 2*x
def third(x):
    return ((x**3) + x**2 + 5*x)
def second(x):
    return ((x**2) + x*2 + 5)
def h(x):
    return [1,2]


print("Creating training data")

def createData(iterations):
    data = []
    for x in tqdm(range(iterations)):
        if x:
            data.append([x, third(x)])
    return Matrix(*data)


TRAIN = createData(ITERATIONS)

print("Created training data")
