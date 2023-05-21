from Python_Linear_Algebra.main import Matrix, Vector
from setup import MAXAGRESSION
from tqdm import tqdm
import time

# only used for creating data or sigmoid funtion. Nothing else.
import math, random

print("Training network")


def getRandom(weight):
    randomNum = random.randint(-weight, weight)
    if randomNum:
        return randomNum
    else:
        return getRandom(weight)


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
        newVal = 1 / (1 + math.exp(-1 * val))
        newActivation.append(newVal)
    return Vector(*newActivation)


def calcLayer(layer: Vector, weights: Matrix) -> Vector:
    return weights * layer


def calcCost(activation: Vector, expected: Vector) -> dict[Vector, Vector]:
    costVal = []
    costDer = []

    for i, val in enumerate(activation):
        costVal.append(
            ((val - expected[i]) ** 2) / expected[i]
            if expected[i]
            else ((val - expected[i]) ** 2)
        )
        costDer.append(((val - expected[i]) * 2))

    return {"value": Vector(*costVal), "derivative": Vector(*costDer)}


def calcNewWeight(cost: Vector, weights: Matrix) -> Matrix:
    newWeights = []

    for r, row in enumerate(weights.rows):
        newRow = []
        if cost["derivative"][r] == 0:
            return Matrix(*weights)
        change = (cost["value"][r]) / (cost["derivative"][r]) * MAXAGRESSION
        for val in row:
            newRow.append((val - change))
        newWeights.append(newRow)

    return Matrix(*newWeights)


def calcAccuracy(output: Vector, expectedOutput: Vector):
    expected = abs(expectedOutput)
    out = abs(output)
    accuracy = 100 * (1 - ((abs(expected - out)) / (expected)))
    return accuracy


class Network:
    # will take in tuple with amount of nodes for each layer
    def __init__(self, setup: tuple, randomWeightRange: int):
        self.layers = {}
        self.weights = {}
        for i, layer in enumerate(setup):
            if i == 0:
                self.layers[0] = Vector()
            else:
                self.weights[i] = Matrix(
                    *[
                        [getRandom(randomWeightRange) for _ in range(setup[i - 1])]
                        for _ in range(layer)
                    ]
                )
                self.layers[i] = Vector()

    def run(self, input: Vector) -> Vector:
        output: int
        lastLayer = len(self.layers) - 1

        for layer in self.layers:
            if layer == 0:
                self.layers[layer] = input
            elif layer == lastLayer:
                self.layers[layer] = calcLayer(
                    self.layers[layer - 1], self.weights[layer]
                )
                output = self.layers[layer]
            else:
                self.layers[layer] = relu(
                    calcLayer(self.layers[layer - 1], self.weights[layer])
                )

        return output

    def __backProp(self, betterLayer: Vector, layerIndex: int):
        cost = calcCost(self.layers[layerIndex], betterLayer)
        self.weights[layerIndex] = calcNewWeight(cost, self.weights[layerIndex])
        if layerIndex != 1:
            nextLayer = relu(calcLayer(betterLayer, self.weights[layerIndex].T()))
            self.__backProp(nextLayer, layerIndex - 1)

    def train(self, trainingData: list[Vector, Vector], delay=0):
        layers = len(self.layers)
        accuracy = 0
        training = tqdm(trainingData)
        for data in training:
            input = data[0]

            expectedOutput = data[1]
            self.run(input)

            # back propigation using for loop
            betterLayer = expectedOutput
            for layerI in self.layers:
                layerIndex = layers - layerI - 1
                if layerIndex != 0:
                    cost = calcCost(self.layers[layerIndex], betterLayer)
                    self.weights[layerIndex] = calcNewWeight(
                        cost, self.weights[layerIndex]
                    )
                    nextLayer = relu(
                        calcLayer(betterLayer, self.weights[layerIndex].T())
                    )
                    betterLayer = nextLayer

            # back propigation using recursive funtion
            # self.__backProp(expectedOutput, layers - 1)

            accuracy = calcAccuracy(self.layers[layers - 1], expectedOutput)
            training.set_postfix({"Accuracy": accuracy})

            if delay:
                time.sleep(delay)
