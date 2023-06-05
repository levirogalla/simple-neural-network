from network.helper.functions import Tested, Activation
from network.nnv1 import HelperFuntions as hfv1
from Python_Linear_Algebra.main import Vector, Matrix
from setup import TRAIN
from tqdm import tqdm


class FourLayer:
    # this network has four layers

    def __init__(self) -> None:
        self.layer1 = Vector()
        self.weights12 = Matrix(
            [Tested.getRandom()], [Tested.getRandom()], [Tested.getRandom()]
        )
        self.layer2 = Vector()
        self.weights23 = Matrix(
            [Tested.getRandom(), Tested.getRandom(), Tested.getRandom()],
            [Tested.getRandom(), Tested.getRandom(), Tested.getRandom()],
            [Tested.getRandom(), Tested.getRandom(), Tested.getRandom()],
        )
        self.layer3 = Vector()
        self.weights34 = Matrix(
            [Tested.getRandom(), Tested.getRandom(), Tested.getRandom()]
        )
        self.layer4 = Vector()

    def train(self):
        for data in tqdm(TRAIN):
            val1 = data[0]
            val2 = data[1]

            self.layer1 = val1
            self.layer2 = Activation.relu(Tested.calcLayer(self.layer1, self.weights12))
            self.layer3 = Activation.relu(Tested.calcLayer(self.layer2, self.weights23))
            self.layer4 = Tested.calcLayer(self.layer3, self.weights34)

            cost43 = hfv1.calcCost(self.layer4, val2)
            self.weights34 = hfv1.calcNewWeight(cost43, self.weights34)
            betterLayer3 = Activation.relu(Tested.calcLayer(val2, self.weights34.T()))

            cost32 = hfv1.calcCost(self.layer3, betterLayer3)
            self.weights23 = hfv1.calcNewWeight(cost32, self.weights23)
            betterLayer2 = Activation.relu(
                Tested.calcLayer(betterLayer3, self.weights23.T())
            )

            cost21 = hfv1.calcCost(self.layer2, betterLayer2)
            self.weights12 = hfv1.calcNewWeight(cost21, self.weights12)

        print(f"Final test: {self.layer1}")
        print(f"Weights from 1 to 2: {self.weights12}")
        print(f"Layer 2: {self.layer2}")
        print(f"Weights from 2 to 3: {self.weights23}")
        print(f"Layer 3: {self.layer3}")
        print(f"Weights from 3 to 4: {self.weights23}")
        print(f"Output: {self.layer4}")


class ThreeLayer:
    # this network has four layers
    def __init__(self) -> None:
        self.layer1 = Vector()
        self.weights12 = Matrix([Tested.getRandom()], [Tested.getRandom()])
        self.layer2 = Vector()
        self.weights23 = Matrix([Tested.getRandom(), Tested.getRandom()])
        self.layer3 = Vector()

    def train(self):
        for data in tqdm(TRAIN):
            val1 = data[0]
            val2 = data[1]

            # forward propigation
            self.layer1 = val1
            self.layer2 = Activation.relu(Tested.calcLayer(self.layer1, self.weights12))
            self.layer3 = Tested.calcLayer(self.layer2, self.weights23)

            # back propigation
            cost32 = hfv1.calcCost(self.layer3, val2)
            self.weights23 = hfv1.calcNewWeight(cost32, self.weights23)
            betterLayer2 = Activation.relu(Tested.calcLayer(val2, self.weights23.T()))

            cost21 = hfv1.calcCost(self.layer2, betterLayer2)
            self.weights12 = hfv1.calcNewWeight(cost21, self.weights12)

        accuracy = Tested.calcAccuracy(self.layer3, val2)
        print(f"Old: {accuracy}")
        return accuracy

        # print(f"Final test: {self.layer1}")
        # print(f"Weights from 1 to 2: {self.weights12}")
        # print(f"Layer 2: {self.layer2}")
        # print(f"Weights from 2 to 3: {self.weights23}")
        # print(f"Layer 3: {self.layer3}")
