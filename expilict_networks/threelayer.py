from Python_Linear_Algebra.main import Matrix, Vector
from network.helper.functions import Tested
from setup import TRAIN, MAXAGRESSION
import math


def calcAccuracy(output: Vector, expectedOutput: Vector):
    expected = abs(expectedOutput)
    out = abs(output)
    accuracy = 100 * (1 - ((abs(expected - out)) / ((expected) if expected else 1)))
    return accuracy


class ThreeLayer:
    def __init__(self) -> None:
        self.layer1 = Vector()
        self.weights12 = Matrix(
            [Tested.getRandom()],
            [Tested.getRandom()],
            [Tested.getRandom()],
            [Tested.getRandom()],
        )
        self.layer2 = Vector()
        self.weights23 = Matrix(
            [
                Tested.getRandom(),
                Tested.getRandom(),
                Tested.getRandom(),
                Tested.getRandom(),
            ]
        )
        self.layer3 = Vector()

    print("Training network")

    def leakyrelu(activation: Vector) -> Vector:
        newActivation = []
        for val in activation:
            if val < 0:
                newActivation.append(val * 0.01)
            else:
                newActivation.append(val)
        return Vector(*newActivation)

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
            newVal = 1 / (1 + math.exp(-val))
            newActivation.append(newVal)
        return Vector(*newActivation)

    def calcLayer(layer: Vector, weights: Matrix) -> Vector:
        newVector = weights * layer
        return newVector

    def calcCostVector(activation: Vector, expected: Vector) -> dict[Vector, Vector]:
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

    def calcCostVal(val: int, expected: int) -> dict[int, int]:
        cost = {}
        cost["value"] = (
            ((val - expected) ** 2) / expected if expected else ((val - expected) ** 2)
        )
        cost["derivative"] = (val - expected) * 2
        return cost

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

                signifiganceBasedOnWeight = (
                    (weight / totalWeight1) if totalWeight1 else 1
                )
                newRow.append((weight - (change * signifiganceBasedOnNodes)))
            newWeights.append(newRow)

        return Matrix(*newWeights)

    def adjustAgression(output, expected):
        totalCost = (expected - output) ** 2
        costDer = (expected - output) * 2
        agression = totalCost / costDer if costDer else 0.00001
        return agression**2 if abs(agression) < MAXAGRESSION else MAXAGRESSION

    # returns how much the weight should be changed based on how far the resulting value was from the expected
    # parameters are the row index, the max change, and the expected value

    def calcSpecificWeightErrorContribution(outputContribution, expectedOutput):
        diff = outputContribution - expectedOutput
        adjustmentFactor = (-1 / (diff**2 + 1)) + 1
        return adjustmentFactor

    def calcChange(cost: dict, multipliers: float = 1):
        maxChange = (
            ((cost["value"] / cost["derivative"]) * multipliers)
            if cost["derivative"] != 0
            else 0
        )
        return maxChange

    def backProp(self, finalExpected: Vector):
        for r, val in enumerate(self.layer3):
            expectedVal = finalExpected[r]
            cost = ThreeLayer.calcCostVal(val, expectedVal)
            weightSignifigance = 1 / len(self.weights23[r])
            maxChange = ThreeLayer.calcChange(cost)

            for c, weight in enumerate(self.weights23[r]):
                outputContribution = weight * self.layer2[c]
                changeAdjustmentFactor = ThreeLayer.calcSpecificWeightErrorContribution(
                    outputContribution, expectedVal
                )
                change = maxChange * changeAdjustmentFactor
                self.weights23[r, c] = weight - change

        betterLayer2 = ThreeLayer.relu(self.weights23.T() * finalExpected)

        for r, val in enumerate(self.layer2):
            expectedVal = betterLayer2[r]
            cost = ThreeLayer.calcCostVal(val, expectedVal)
            weightSignifigance = 1 / len(self.weights12[r])
            maxChange = ThreeLayer.calcChange(cost)

            for c, weight in enumerate(self.weights12[r]):
                outputContribution = weight * self.layer1[c]
                changeAdjustmentFactor = ThreeLayer.calcSpecificWeightErrorContribution(
                    outputContribution, expectedVal
                )
                change = maxChange * changeAdjustmentFactor
                self.weights12[r, c] = weight - change

    def train(self):
        file = open("data/data2.txt", "w")
        for i, data in enumerate(TRAIN):
            val1 = data[0]
            val2 = data[1]

            self.layer1 = val1
            file.write(f"{self.layer1}")
            file.write(f"{self.weights12}")

            self.layer2 = ThreeLayer.relu(
                ThreeLayer.calcLayer(self.layer1, self.weights12)
            )
            file.write(f"{self.layer2}")
            file.write(f"{self.weights23}")

            self.layer3 = ThreeLayer.calcLayer(self.layer2, self.weights23)
            file.write(f"{self.layer3}\n\n")

            self.backProp(val2)
            # newWeights = []
            # for r, val in enumerate(self.layer3):
            #     newRow = []
            #     expectedVal = val2[r]
            #     cost = ThreeLayer.calcCostVal(val, expectedVal)
            #     weightSignifigance = 1 / len(self.weights23[r])
            #     maxChange = (
            #         (
            #             (cost["value"] / cost["derivative"])
            #         )  # <- goes in the bracket* weightSignifigance
            #         if cost["derivative"] != 0
            #         else 0
            #     )

            #     for c, weight in enumerate(self.weights23[r]):
            #         outputContribution = weight * self.layer2[c]
            #         changeAdjustmentFactor = (
            #             ThreeLayer.calcSpecificWeightErrorContribution(
            #                 outputContribution, expectedVal
            #             )
            #         )
            #         change = maxChange * changeAdjustmentFactor
            #         newWeight = weight - change
            #         self.weights23[r, c] = newWeight

            #         newRow.append(newWeight)
            #     newWeights.append(newRow)

            # self.weights23 = Matrix(*newWeights)

            # # print(test)
            # print(self.weights23)
            # print(self.weights23.T())

            # betterLayer2 = ThreeLayer.relu(self.weights23.T() * val2)

            # print(betterLayer2, "\n\n")

            # newWeights = []
            # for r, val in enumerate(self.layer2):
            #     newRow = []
            #     expectedVal = betterLayer2[r]
            #     cost = ThreeLayer.calcCostVal(val, expectedVal)
            #     weightSignifigance = 1 / len(self.weights12[r])
            #     maxChange = (
            #         (
            #             (cost["value"] / cost["derivative"])
            #         )  # <- goes in the bracket* weightSignifigance
            #         if cost["derivative"] != 0
            #         else 0
            #     )

            #     for c, weight in enumerate(self.weights12[r]):
            #         outputContribution = weight * self.layer1[c]
            #         changeAdjustmentFactor = (
            #             ThreeLayer.calcSpecificWeightErrorContribution(
            #                 outputContribution, expectedVal
            #             )
            #         )
            #         change = maxChange * changeAdjustmentFactor
            #         newWeight = weight - change
            #         newRow.append(newWeight)
            #         # self.weights12[r][c] = self.weights12[r][c] - change
            #     newWeights.append(newRow)
            # self.weights12 = Matrix(*newWeights)

        accuracy = calcAccuracy(self.layer3, val2)
        print(f"New: {accuracy}")
        file.close()
        return accuracy
        # print(f"Final test: {self.layer1}")
        # print(f"Weights from 1 to 2: {self.weights12}")
        # print(f"Layer 2: {self.layer2}")
        # print(f"Weights from 2 to 3: {self.weights23}")
        # print(f"Final output: {self.layer3}")
