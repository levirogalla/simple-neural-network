from Python_Linear_Algebra.main import Matrix, Vector
from setup import MAXAGRESSION 
import math

print("Training network")

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

