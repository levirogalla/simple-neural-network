from Python_Linear_Algebra.main import Matrix, Vector
import random

TRAIN = Matrix(
    [1,2],
    [100,200],
    [2,4],
    [3,6],
    [4,8],
    [5,10]
)

layer1 = Vector()
layer2 = Vector()
weights12 = Matrix([random.randint(-10,10)])

def calcLayer2(layer1: Vector, weights: Matrix) -> Vector:
    return  weights * layer1

def calcCost(activation: Vector, expected: int) -> dict[Vector, Vector]:
    costVal = []
    costDer = []
    for val in activation:
        costVal.append((val-expected)**2)
        costDer.append((val-expected)*2)
    return {
        'value': Vector(*costVal),
        'derivative': Vector(*costDer)
    }

def calcNewWeight(cost: Vector, weights: Matrix) -> Matrix:
    newWeights = []
    for r, row in enumerate(weights.rows):
        newRow = []
        totalW = abs(row)
        change = (cost['value'][r])/(cost['derivative'][r])
        for val in row:
            # if makes sure not to divide by 0, only really mater for single node layers
            weightImportanceFactor = abs((val/totalW)) if totalW else 0.01
            newRow.append(
                weightImportanceFactor*(val-change)
            )
        newWeights.append(newRow)

    return Matrix(*newWeights)

for i, data in enumerate(TRAIN):
    print(f"Weights are: {weights12}")

    # normalizing dataset
    val1 = data[0]/data[0]
    val2 = data[1]/data[0]

    layer1 = Vector(val1)
    output = calcLayer2(layer1, weights12)

    print(f"Output is: {output}")

    cost = calcCost(output, val2)

    if abs(cost['derivative']) == 0:
        print("Model is optimized")
        break

    weights12 = calcNewWeight(cost, weights12)



print(layer1)
print(weights12)
