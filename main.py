from network.nnv1 import Network as nnv1
from network.nnv2 import Network as nnv2
from expilict_networks.threelayer import ThreeLayer as New
from expilict_networks.fourlayer import ThreeLayer as Old
import time
from Python_Linear_Algebra.main import Matrix, Vector
from tqdm import tqdm
from setup import TRAIN

# x = nnv2((1, 10, 1), 10, "leakyrelu")

# x.train(
#     TRAIN,
#     progressBar=True,
#     logToFile="data/data4.txt",
#     useBasicWeightSignifigance=False,
# )

# print(x.run(Vector(0)))
# print(x.run(Vector(1)))
# print(x.run(Vector(2)))


# print(Matrix([2.642, -1.442]).inverse())

# run = True

# while run:
#     xVal = int(input("Input a number: "))

#     print(f"Neural network produced: {x.run(Vector(xVal))}")

#     terminate = input("Terminate program (Y/n): ")

#     if terminate == "Y":
#         run = False


def Test():
    TESTS = 1000

    accuracyTotalA = 0
    trainingTimeTotalA = 0

    accuracyTotalB = 0
    trainingTimeTotalB = 0

    for _ in tqdm(range(TESTS)):
        x = nnv2((1, 2, 1), 10, "relu")
        st = time.process_time_ns()
        accuracyTotalA += x.train(TRAIN, test=True)
        end = time.process_time_ns()
        trainingTimeTotalA += end - st

        y = nnv2((1, 2, 1), 10, "relu")
        st = time.process_time_ns()
        accuracyTotalB += y.train(TRAIN, test=False)
        end = time.process_time_ns()
        trainingTimeTotalB += end - st

    print(
        f"\nAfter {TESTS} tests A has an avg accuracy of {accuracyTotalA/TESTS} and avg training time of {trainingTimeTotalA/TESTS}s\n"
    )
    print(
        f"\nAfter {TESTS} tests B has an avg accuracy of {accuracyTotalB/TESTS} and avg training time of {trainingTimeTotalB/TESTS}s\n"
    )


Test()
