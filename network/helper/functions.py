from Python_Linear_Algebra.main import Matrix, Vector
from setup import MAXAGRESSION
import random, math


class Activation:
    def relu(activation: Vector) -> Vector:
        newActivation = []
        for val in activation:
            if val < 0:
                newActivation.append(0)
            else:
                newActivation.append(val)
        return Vector(*newActivation)

    def leakyrelu(activation: Vector) -> Vector:
        newActivation = []
        for val in activation:
            if val < 0:
                newActivation.append(val * 0.01)
            else:
                newActivation.append(val)
        return Vector(*newActivation)

    def sigmoid(activation: Vector) -> Vector:
        newActivation = []
        for val in activation:
            newVal = 1 / (1 + math.exp(-1 * val))
            newActivation.append(newVal)
        return Vector(*newActivation)


class Tested:
    def getRandom(weight):
        randomNum = random.randint(-weight, weight)
        if randomNum:
            return randomNum
        else:
            return Tested.getRandom(weight)

    def calcLayer(layer: Vector, weights: Matrix) -> Vector:
        nextLayer = weights * layer
        return nextLayer

    def calcAccuracy(output: Vector, expectedOutput: Vector):
        expected = abs(expectedOutput)
        out = abs(output)
        accuracy = 100 * (1 - ((abs(expected - out)) / ((expected) if expected else 1)))
        return accuracy


class Testing:
    pass
