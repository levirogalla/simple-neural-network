from Python_Linear_Algebra.main import Matrix, Vector
from setup import TRAIN
import random

print("Training network")

AGRESSION = 0.1

# one node
layer1 = Vector()
weights12 = Matrix([2], [2])
# two nodes
layer2 = Vector()
weights23 = Matrix([2, 2])
# one node
layer3 = Vector()



def calcLayer(layer: Vector, weights: Matrix) -> Vector:
    return  weights * layer

def calcCost(activation: Vector, expected: Vector) -> dict[Vector, Vector]:
    costVal = []
    costDer = []

    for i, val in enumerate(activation):
        costVal.append(((val-expected[i])**2)/expected[i])
        costDer.append(((val-expected[i])*2))

    return {
        'value': Vector(*costVal),
        'derivative': Vector(*costDer)
    }

def calcNewWeight(cost: Vector, weights: Matrix) -> Matrix:
    newWeights = []

    for r, row in enumerate(weights.rows):
        newRow = []
        if cost['derivative'][r] == 0:
            return Matrix(*weights)
        change = (cost['value'][r])/(cost['derivative'][r])*AGRESSION
        for val in row:
            newRow.append(
                (val-change)
            )
        newWeights.append(newRow)

    return Matrix(*newWeights)

file = open("data.txt", "w")
for i, data in enumerate(TRAIN):

    # normalizing dataset
    val1 = data[0] 
    val2 = data[1]

    print("\nCalculating layers")
    layer1 = Vector(val1)
    file.write(f"{layer1}\n")
    print(f"layer1: {layer1}")
    print(f"weights from 1 to 2: {weights12}")
    layer2 = calcLayer(layer1, weights12)
    file.write(f"{layer2}\n")
    print(f"layer2: {layer2}")
    print(f"weights from 2 to 3: {weights23}")
    layer3 = calcLayer(layer2, weights23)
    file.write(f"{layer3}\n\n")
    print(f"layer3: {layer3}")
 

    print("\nCalculating weights and cost")
    cost32 = calcCost(layer3, Vector(val2))
    print(f"the cost from 3 to 2: {cost32}")

    weights23 = calcNewWeight(cost32, weights23)
    print(f"new weights from 2 to 3: {weights23}")
    betterLayer2 = (weights23.T())*(layer3)
    cost21 = calcCost(layer2, betterLayer2)
    print(f"the cost from 2 to 1: {cost21}")
    weights12 = calcNewWeight(cost21, weights12)
    print(f"new weights from 1 to 2: {weights12}\n\n")


print(f"Final test: {layer1}")
print(weights12)
print(layer2)
print(weights23)
print(f"Final output: {layer3}")
file.close()