from Python_Linear_Algebra.main import Matrix, Vector
from setup import MAXAGRESSION
from network.helper.functions import Activation, Tested
from tqdm import tqdm
import time

# only used for creating data or sigmoid funtion. Nothing else.
import math, random


class HelperFuntion:
    def calcCostVal(val: int, expected: int) -> dict[int, int]:
        cost = {}
        cost["value"] = (val - expected) ** 2  # note: took away divide by expected
        cost["derivative"] = (val - expected) * 2
        return cost

    def calcSpecificWeightErrorContribution(
        outputContribution, expectedOutput
    ) -> float:
        diff = outputContribution - expectedOutput
        adjustmentFactor = (-1 / (diff**2 + 1)) + 1
        return adjustmentFactor

    def calcChange(cost: dict, multipliers: float = 1) -> float:
        maxChange = (
            ((cost["value"] / cost["derivative"]) * multipliers)
            if cost["derivative"] != 0
            else 0
        )
        return maxChange


class Network:
    # will take in tuple with amount of nodes for each layer
    def __init__(self, setup: tuple, randomWeightRange: int, activation: str):
        self.layers: list[Vector] = []
        self.weights: list[Matrix] = []
        for i, layer in enumerate(setup):
            if i == 0:
                self.layers.append(Vector())
            else:
                self.weights.append(
                    Matrix(
                        *[
                            [
                                Tested.getRandom(randomWeightRange)
                                for _ in range(setup[i - 1])
                            ]
                            for _ in range(layer)
                        ]
                    )
                )
                self.layers.append(Vector())

        if activation == "relu":
            self.activation = Activation.relu
            self.activationInverse = Activation.reluInverse
        if activation == "leakyrelu":
            self.activation = Activation.leakyrelu
            self.activationInverse = Activation.leakyreluInverse
        if activation == "sigmoid":
            self.activation = Activation.sigmoid
            self.activation = Activation.sigmoidInverse

    def run(self, input: Vector, logToFile: str = None) -> Vector:
        output: Vector
        lastLayerIndex = len(self.layers) - 1
        if logToFile is not None:
            file = open(logToFile, "a")

        # looping through layers in model object to calculate next layer
        for i, _ in enumerate(self.layers):
            # condition where the layer is the input layer
            if i == 0:
                self.layers[i] = input

            # condition where the layer is the output layer, doesnt apply actiavtion funtion to ouptu layer
            elif i == lastLayerIndex:
                self.layers[i] = Tested.calcLayer(
                    self.layers[i - 1], self.weights[i - 1]
                )
                output = self.layers[i]

            # condition for hidden layers
            else:
                self.layers[i] = self.activation(
                    Tested.calcLayer(self.layers[i - 1], self.weights[i - 1])
                )

            # writes all layeres and weight to a file, may change this to right to a pickle file in futer to perserve object when train is interupted
            if logToFile is not None:
                file.write(f"{self.layers[i]}")
                if i != lastLayerIndex:
                    file.write(f"{self.weights[i]}")

        if logToFile is not None:
            file.write("\n\n")
            file.close()

        return output

    def backProp(
        self,
        finalExpected: Vector,
        useBasicWeightSignifigance: bool,
        agression: float = 1,
    ):
        layers = len(self.layers)
        betterLayer = finalExpected
        for layerI, _ in enumerate(self.layers):
            # needs to go through the indexs backwards
            layerIndex = layers - layerI - 1

            if layerIndex != 0:
                for r, val in enumerate(self.layers[layerIndex]):
                    expectedVal = betterLayer[r]
                    cost = HelperFuntion.calcCostVal(val, expectedVal)
                    weightSignifigance = (
                        1 / len(self.weights[layerIndex - 1][r])
                        if useBasicWeightSignifigance
                        else 1
                    )

                    maxChange = (
                        (
                            (HelperFuntion.calcChange(cost, weightSignifigance))
                            / (sum(self.layers[layerIndex - 1]) + 0.001)
                        )
                        if self.layers[layerIndex - 1] != 0
                        else 0
                    )  # normalizes cost by dividing by the sum (or abs?)

                    for c, weight in enumerate(self.weights[layerIndex - 1][r]):
                        outputContribution = weight * self.layers[layerIndex - 1][c]
                        changeAdjustmentFactor = (
                            HelperFuntion.calcSpecificWeightErrorContribution(
                                outputContribution, expectedVal
                            )
                        )
                        change = maxChange * changeAdjustmentFactor * agression
                        self.weights[layerIndex - 1][r, c] = weight - change

                reversedWeights = self.weights[layerIndex - 1].inverse()
                betterLayer = self.activation(reversedWeights * betterLayer)

    def train(
        self,
        trainingData: list[Vector, Vector],
        delay=0,  # adds a delay to the end of each training cycle
        logToFile: str = None,  # if the weights and activations should be logges to a file every training iterations
        stopAt: tuple[
            int, int
        ] = None,  # what accuracy the network should stop at, first index is accurcy val and second is percision
        progressBar: bool = False,  # whether or not to use a progress bar when training
        useBasicWeightSignifigance: bool = False,  # this is used for back prop to limit the amount a weight can be changed based on how many nodes there are
        agression: float = 1,
    ):
        layers = len(self.layers)
        accuracy = 0
        trainingData: tqdm

        if logToFile is not None:
            # clears file
            open(logToFile, "w").close()

        if progressBar:
            trainingData = tqdm(trainingData)

        for data in trainingData:
            input = data[0]
            expectedOutput = data[1]
            self.run(input, logToFile=logToFile)

            # back propigation using for loop
            self.backProp(expectedOutput, useBasicWeightSignifigance, agression)

            accuracy = Tested.calcAccuracy(self.layers[layers - 1], expectedOutput)
            if progressBar:
                trainingData.set_postfix({"Accuracy": accuracy})

            # checks stop at first so accuracy isnt rounded every time
            if stopAt is not None:
                if stopAt[0] == round(accuracy, stopAt[1]):
                    break

            if delay:
                time.sleep(delay)

        return accuracy
