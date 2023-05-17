from Python_Linear_Algebra.main import Matrix

def f(x): 
    return ((3*(x**2) - 15) + 2**x)
def g(x):
    return 2*x
def h(x):
    return [1,2]
ITERATIONS = 9999

print("Creating training data")

def createData(iterations):
    data = []
    for x in range(iterations):
        if x:
            data.append(h(x))
    return Matrix(*data)


TRAIN = createData(ITERATIONS)

print("Created training data")
