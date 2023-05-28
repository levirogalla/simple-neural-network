from Python_Linear_Algebra.main import Matrix, Vector
from setup import MAXAGRESSION
from network.helper.functions import Activation, Tested
import time
from tqdm import tqdm

# only used for creating data or sigmoid funtion. Nothing else.
import math, random


class HelperFuntions:
    # previousActivation: Vector, expected: Vector
    def calcNewWeight(cost: Vector, weights: Matrix) -> Matrix:
        newWeights = []

        for r, row in enumerate(weights.rows):
            newRow = []
            if cost["derivative"][r] == 0:
                return Matrix(*weights)
            change = (cost["value"][r]) / (cost["derivative"][r]) * MAXAGRESSION
            signifiganceBasedOnNodes = 1 / len(row)
            for weight in row:
                totalWeight1 = abs(row)
                # totalWeight2 = sum of all weights in row

                signifiganceBasedOnWeight = 0
                newRow.append((weight - (change * signifiganceBasedOnNodes)))
            newWeights.append(newRow)

        return Matrix(*newWeights)

    def calcCost(activation: Vector, expected: Vector) -> dict[Vector, Vector]:
        costVal = []
        costDer = []

        for i, val in enumerate(activation):
            costVal.append(
                # divided by expected to normalize change
                ((val - expected[i]) ** 2) / expected[i]
                if expected[i]
                else ((val - expected[i]) ** 2)
            )
            costDer.append(((val - expected[i]) * 2))

        return {"value": Vector(*costVal), "derivative": Vector(*costDer)}


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
                        [
                            Tested.getRandom(randomWeightRange)
                            for _ in range(setup[i - 1])
                        ]
                        for _ in range(layer)
                    ]
                )
                self.layers[i] = Vector()

    def run(self, input: Vector, logToFile: str = None) -> Vector:
        output: int
        lastLayer = len(self.layers) - 1

        if logToFile is not None:
            file = open(logToFile, "a")
        for layer in self.layers:
            if layer == 0:
                self.layers[layer] = input
            elif layer == lastLayer:
                self.layers[layer] = Tested.calcLayer(
                    self.layers[layer - 1], self.weights[layer]
                )
                output = self.layers[layer]

            # condition for hidden layers
            else:
                self.layers[layer] = Activation.relu(
                    Tested.calcLayer(self.layers[layer - 1], self.weights[layer])
                )
            if logToFile is not None:
                if layer != 0:
                    file.write(f"{self.weights[layer]}")
                file.write(f"{self.layers[layer]}")

        if logToFile is not None:
            file.write("\n\n")
            file.close()

        return output

    def __backProp(self, betterLayer: Vector, layerIndex: int):
        cost = HelperFuntions.calcCost(self.layers[layerIndex], betterLayer)
        self.weights[layerIndex] = HelperFuntions.calcNewWeight(
            cost, self.weights[layerIndex]
        )
        if layerIndex != 1:
            nextLayer = Activation.relu(
                Tested.calcLayer(betterLayer, self.weights[layerIndex].T())
            )
            self.__backProp(nextLayer, layerIndex - 1)

    def train(
        self,
        trainingData: list[Vector, Vector],
        delay=0,
        logToFile: str = None,
        stopAt: tuple[int, int] = None,
    ):
        layers = len(self.layers)
        accuracy = 0

        if logToFile is not None:
            # clears file
            open(logToFile, "w").close()

        training = tqdm(trainingData)
        for data in training:
            input = data[0]
            expectedOutput = data[1]
            self.run(input, logToFile=logToFile)

            # back propigation using for loop
            betterLayer = expectedOutput
            for layerI, _ in enumerate(self.layers):
                # needs to go through the indexs backwards
                layerIndex = layers - layerI - 1

                if layerIndex != 0:
                    cost = HelperFuntions.calcCost(self.layers[layerIndex], betterLayer)
                    self.weights[layerIndex] = HelperFuntions.calcNewWeight(
                        cost, self.weights[layerIndex]
                    )
                    nextLayer = Activation.relu(
                        Tested.calcLayer(betterLayer, self.weights[layerIndex].T())
                    )
                    betterLayer = nextLayer

            # back propigation using recursive funtion
            # self.__backProp(expectedOutput, layers - 1)

            accuracy = Tested.calcAccuracy(self.layers[layers - 1], expectedOutput)
            training.set_postfix({"Accuracy": accuracy})

            # checks stop at first so accuracy isnt rounded every time
            if stopAt is not None:
                if stopAt[0] == round(accuracy, stopAt[1]):
                    break

            if delay:
                time.sleep(delay)
        return accuracy
