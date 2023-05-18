from network import calcLayer, calcCost, calcNewWeight, relu
from Python_Linear_Algebra.main import Vector, Matrix
from setup import TRAIN, getRandom
from tqdm import tqdm

class FourLayer:
     # this network has four layers

    def __init__(self) -> None:
        self.layer1 = Vector()
        self.weights12 = Matrix([getRandom()], [getRandom()])
        self.layer2 = Vector()
        self.weights23 = Matrix([getRandom(), getRandom()], [getRandom(), getRandom()])
        self.layer3 = Vector()
        self.weights34 = Matrix([getRandom(), getRandom()])
        self.layer4 = Vector()

    def train(self):

        for data in tqdm(TRAIN):
            val1 = Vector(data[0])
            val2 = Vector(data[1])

            self.layer1=val1
            self.layer2=relu(calcLayer(self.layer1, self.weights12))
            self.layer3=relu(calcLayer(self.layer2, self.weights23))
            self.layer4=calcLayer(self.layer3, self.weights34)

            cost43 = calcCost(self.layer4, val2)
            self.weights34 = calcNewWeight(cost43, self.weights34)
            betterLayer3 = relu(calcLayer(val2, self.weights34.T()))

            cost32 = calcCost(self.layer3, betterLayer3)
            self.weights23 = calcNewWeight(cost32, self.weights23)
            betterLayer2 = relu(calcLayer(betterLayer3, self.weights23.T()))

            cost21 = calcCost(self.layer2, betterLayer2)
            self.weights12 = calcNewWeight(cost21, self.weights12)

        print(f"Final test: {self.layer1}")
        print(f"Weights from 1 to 2: {self.weights12}")
        print(f"Layer 2: {self.layer2}")
        print(f"Weights from 2 to 3: {self.weights23}")
        print(f"Layer 3: {self.layer3}")
        print(f"Weights from 3 to 4: {self.weights23}")
        print(f"Output: {self.layer4}")

        print(f"Finished with accuracy of: {self.layer4[0]/TRAIN[-1][1]}")

class ThreeLayer:
    # this network has four layers
    def __init__(self) -> None:    
        self.layer1 = Vector()
        self.weights12 = Matrix(
            [getRandom()], [getRandom()], [getRandom()], [getRandom()], [getRandom()], [getRandom()]
        )
        self.layer2 = Vector()
        self.weights23 = Matrix(
            [getRandom(), getRandom(), getRandom(), getRandom(), getRandom(), getRandom()]
        )
        self.layer3 = Vector()

    def train(self):
        for data in tqdm(TRAIN):
            val1 = Vector(data[0])
            val2 = Vector(data[1])

            # forward propigation
            self.layer1 = val1
            self.layer2 = relu(calcLayer(self.layer1, self.weights12))
            self.layer3 = calcLayer(self.layer2, self.weights23)

            # back propigation
            cost32 = calcCost(self.layer3, val2)
            self.weights23 = calcNewWeight(cost32, self.weights23)
            betterLayer2 = relu(calcLayer(val2, (self.weights23.T())))

            cost21 = calcCost(self.layer2, betterLayer2)
            self.weights12 = calcNewWeight(cost21, self.weights12)

        print(f"Final test: {self.layer1}")
        print(f"Weights from 1 to 2: {self.weights12}")
        print(f"Layer 2: {self.layer2}")
        print(f"Weights from 2 to 3: {self.weights23}")
        print(f"Layer 3: {self.layer3}")
    
        print(f"Finished with accuracy of: {self.layer3[0]/TRAIN[-1][1]}")



FourLayer().train()

ThreeLayer().train()