from Python_Linear_Algebra.main import Matrix, Vector
import random


ITERATIONS = 100
MAXAGRESSION = 1
INITWEIGHT = 1

def getRandom(): 
    randomNum = random.randint(-10,10) 
    if randomNum:
        return randomNum
    else:
        return getRandom()
    

# one node
layer1 = Vector()
weights12 = Matrix([getRandom()], [getRandom()])
# two nodes
layer2 = Vector()
weights23 = Matrix([getRandom(), getRandom()])
# one node
layer3 = Vector()


def f(x): 
    return ((3*(x**2) - 15) + 2**x)
def g(x):
    return 2*x
def n(x):
    return ((x**2) + 5)
def h(x):
    return [1,2]

print("Creating training data")

def createData(iterations):
    data = []
    for x in range(iterations):
        if x:
            data.append([x, n(x)])
    return Matrix(*data)


TRAIN = createData(ITERATIONS)

print("Created training data")
