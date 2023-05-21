from Python_Linear_Algebra.main import Matrix, Vector
from setup import TRAIN
import random


# this network only works with linear function and train data has to be integers not vectores
class TwoLayer:
    print("Training network")

    layer1 = Vector()
    layer2 = Vector()
    weights12 = Matrix([random.randint(-10, 10)])

    def calcLayer2(layer1: Vector, weights: Matrix) -> Vector:
        return weights * layer1

    def calcCost(activation: Vector, expected: Vector) -> dict[Vector, Vector]:
        costVal = []
        costDer = []
        for i, val in enumerate(activation):
            costVal.append((val - expected[i]) ** 2)
            costDer.append((val - expected[i]) * 2)
        return {"value": Vector(*costVal), "derivative": Vector(*costDer)}

    def calcNewWeight(cost: Vector, weights: Matrix) -> Matrix:
        newWeights = []
        for r, row in enumerate(weights.rows):
            newRow = []
            totalW = abs(row)
            change = (cost["value"][r]) / (cost["derivative"][r])
            for val in row:
                # if makes sure not to divide by 0, only really mater for single node layers
                weightImportanceFactor = abs((val / totalW)) if totalW else 0.01
                newRow.append(weightImportanceFactor * (val - change))
            newWeights.append(newRow)

        return Matrix(*newWeights)

    for i, data in enumerate(TRAIN):
        # normalizing dataset
        # this is wrong im just keeping it because this is my first attempt and is funny
        val1 = data[0] / data[0]
        val2 = data[1] / data[0]

        layer1 = Vector(val1)
        output = calcLayer2(layer1, weights12)

        cost = calcCost(output, Vector(val2))

        if abs(cost["derivative"]) == 0:
            print("Model is optimized")
            break

        weights12 = calcNewWeight(cost, weights12)

    print(layer1)
    print(weights12)
