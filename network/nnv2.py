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
        cost["value"] = (val - expected) ** 2
        cost["derivative"] = (val - expected) * 2
        return cost

    def positiveContribution(outputContribution, expectedOutput) -> float:
        percentContribution = outputContribution - expectedOutput
        adjustmentFactor = (-1 / ((percentContribution**2) + 1)) + 1
        return adjustmentFactor

    def negativeContribution(outputContribution, output) -> float:
        percentContribution = outputContribution - output
        adjustmentFactor = 1 / ((percentContribution**2) + 1)
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
        self.baiss: list[Vector] = []
        for i, nodeAmount in enumerate(setup):
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
                            for _ in range(nodeAmount)
                        ]
                    )
                )
                self.baiss.append(Vector(*[0 for _ in range(nodeAmount)]))
                self.layers.append(Vector())

        if activation == "relu":
            self.activation = Activation.relu
            self.activationInverse = Activation.reluInverse
        if activation == "leakyrelu":
            self.activation = Activation.leakyrelu
            self.activationInverse = Activation.leakyreluInverse
        if activation == "sigmoid":
            self.activation = Activation.sigmoid
            self.activationInverse = Activation.sigmoidInverse
        if activation == "linear":
            self.activation = Activation.linear
            self.activationInverse = Activation.linear

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
                    self.layers[i - 1], self.weights[i - 1], self.baiss[i - 1]
                )
                output = self.layers[i]

            # condition for hidden layers
            else:
                self.layers[i] = self.activation(
                    Tested.calcLayer(
                        self.layers[i - 1], self.weights[i - 1], self.baiss[i - 1]
                    )
                )

            # writes all layeres and weight to a file, may change this to right to a pickle file in futer to perserve object when train is interupted
            if logToFile is not None:
                if i != 0:
                    file.write(
                        f"Weights before biases: {self.layers[i] - self.baiss[i-1]}\n"
                    )
                    file.write(f"Biases: {self.baiss[i-1]}\n")
                file.write(f"Weights after Biases: {self.layers[i]}")
                if i != lastLayerIndex:
                    file.write(f"{self.weights[i-1]}")

        if logToFile is not None:
            file.write("\n\n")
            file.close()

        return output

    def backProp(
        self,
        finalExpected: Vector,
        useBasicWeightSignifigance: bool,
        agression: float = 1,
        test=None,
    ):
        layers = len(self.layers)

        # sets the last layer to the expected layer
        betterLayer = finalExpected

        # needs to go through the indexs backwards
        for layerI, _ in enumerate(self.layers):
            layerIndex = layers - layerI - 1

            # loops through all activations of all layers that is not the input layer
            if layerIndex != 0:
                for r, val in enumerate(self.layers[layerIndex]):
                    # calculates the cost between expected activation and current actiavtions
                    expectedVal = betterLayer[r]
                    cost = HelperFuntion.calcCostVal(val, expectedVal)

                    # if use the user chooses, will create a factor of 1 / amount of nodes in layer to avoid overshooting the root of the individual cost of each node
                    weightSignifigance = (
                        1 / len(self.weights[layerIndex - 1][r])
                        if useBasicWeightSignifigance
                        else 1
                    )

                    # applies the wieght signifigance to the change
                    maxChange = (
                        # calculates change with newtons method i.e. function/derivative and normalizes the cost with the sum of activations in the previous layer
                        (HelperFuntion.calcChange(cost, weightSignifigance))
                        / (
                            sum(self.layers[layerIndex - 1]) + 0.000000001
                        )  # prevents division by 0
                    )

                    # loops through all weigths connected to the current node from the previous layer
                    for c, weight in enumerate(self.weights[layerIndex - 1][r]):
                        # gets the contribuiotn of the activation from the previous layer to the current node
                        outputContribution = weight * self.layers[layerIndex - 1][c]

                        # if the contribution was very close the expected value the change adjustment factor will aproach 0
                        # if it is far it will aproach 1
                        postiveReinforcement = HelperFuntion.positiveContribution(
                            outputContribution, expectedVal
                        )

                        # if the contribution was very close the error the change adjustment factor will aproach 1
                        # if it is far it will aproach 0
                        negativeReinforcement = HelperFuntion.negativeContribution(
                            outputContribution, val
                        )
                        # make the above statement true
                        negativeReinforcement = 1 - negativeReinforcement

                        # calculates over all change based on all factors
                        if not test:
                            negativeReinforcement = 1

                        change = (
                            maxChange
                            * postiveReinforcement
                            * negativeReinforcement
                            * agression
                        )

                        # applies change to weight
                        self.weights[layerIndex - 1][r, c] = weight - change

                # with all weight updated this calculates what a better previous layer would look like
                reversedWeights = self.weights[layerIndex - 1].inverse()

                betterLayer = reversedWeights * betterLayer

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
        test=None,
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
            self.backProp(expectedOutput, useBasicWeightSignifigance, agression, test)

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
