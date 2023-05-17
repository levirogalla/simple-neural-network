from Python_Linear_Algebra.main import Matrix, Vector
from setup import TRAIN, MAXAGRESSION 
from setup import layer1, layer2, layer3, weights12, weights23
import math

print("Training network")

agression = 0.1

def relu(activation: Vector) -> Vector:
    newActivation = []
    for val in activation:  
        if val < 0:
            newActivation.append(0)
        else:
            newActivation.append(val)
    return Vector(*newActivation)
    
def sigmoid(activation: Vector) -> Vector:
    newActivation = []
    for val in activation:
        newVal = 1/(1+math.exp(-1*val))
        newActivation.append(newVal)
    return Vector(*newActivation)

def calcLayer(layer: Vector, weights: Matrix) -> Vector:
    return  weights * layer


def calcCost(activation: Vector, expected: Vector) -> dict[Vector, Vector]:
    costVal = []
    costDer = []

    for i, val in enumerate(activation):
        costVal.append(((val-expected[i])**2)/expected[i] if expected[i] else ((val-expected[i])**2))
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
        change = (cost['value'][r])/(cost['derivative'][r])*MAXAGRESSION
        for val in row:
            newRow.append(
                (val-change)
            )
        newWeights.append(newRow)

    return Matrix(*newWeights)

def adjustAgression(output, expected):
    totalCost = (expected - output)**2
    costDer = (expected - output) * 2
    agression = totalCost/costDer if costDer else 0.00001
    return (agression**2 if abs(agression) < MAXAGRESSION else MAXAGRESSION)

file = open("data.txt", "w")
for i, data in enumerate(TRAIN):
    # normalizing dataset
    val1 = data[0] 
    val2 = data[1]

    layer1 = Vector(val1)
    file.write(f"{layer1}")
    file.write(f"{weights12}")

    layer2 = relu(calcLayer(layer1, weights12))
    file.write(f"{layer2}")
    file.write(f"{weights23}")

    layer3 = calcLayer(layer2, weights23)
    file.write(f"{layer3}\n\n")
 
    cost32 = calcCost(layer3, Vector(val2))

    weights23 = calcNewWeight(cost32, weights23)

    # pass val2 or layer 3 into this?
    betterLayer2 = relu((weights23.T())*(val2))
    cost21 = calcCost(layer2, betterLayer2)

    weights12 = calcNewWeight(cost21, weights12)
 


print(f"Final test: {layer1}")
print(f"Weights from 1 to 2: {weights12}")
print(f"Layer 2: {layer2}")
print(f"Weights from 2 to 3: {weights23}")
print(f"Final output: {layer3}")

print(f"Finished with accuracy of: {layer3[0]/TRAIN[-1][1]}")

file.close()

