from Python_Linear_Algebra.main import Matrix, Vector
from setup import MAXAGRESSION
from network.helper.functions import Activation, Tested, getRandom
from tqdm import tqdm
import time

# only used for creating data or sigmoid funtion. Nothing else.
import math, random


class HelperFuntion:
    def calcCostVal(val: int, expected: int) -> dict[int, int]:
        cost = {}
        cost["value"] = (
            ((val - expected) ** 2) / expected if expected else ((val - expected) ** 2)
        )
        cost["derivative"] = (val - expected) * 2
        return cost


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

    def run(self, input: Vector, logToFile=None | str) -> Vector:
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
        cost = Tested.calcCost(self.layers[layerIndex], betterLayer)
        self.weights[layerIndex] = Tested.calcNewWeight(cost, self.weights[layerIndex])
        if layerIndex != 1:
            nextLayer = Activation.relu(
                Tested.calcLayer(betterLayer, self.weights[layerIndex].T())
            )
            self.__backProp(nextLayer, layerIndex - 1)

    def train(
        self,
        trainingData: list[Vector, Vector],
        delay=0,
        logToFile=None | str,
        stopAt=None | tuple[int, int],
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
                    cost = Tested.calcCost(self.layers[layerIndex], betterLayer)
                    self.weights[layerIndex] = Tested.calcNewWeight(
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
