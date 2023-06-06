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
            newVal = 1 / (1 + math.exp(-val))
            newActivation.append(newVal)
        return Vector(*newActivation)


class Tested:
    def getRandom(weight=10):
        randomNum = random.randint(0, weight)
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

    def flatten(inputImgae: list[list]):
        "Takes in 2x2 array of number values and returns a flattened vector"
        new = []
        for row in inputImgae:
            for val in row:
                new.append(val)
        return Vector(*new)


class Testing:
    pass
