from network.nnv1 import Network as nnv1
from setup import TRAIN
from expilict_networks.threelayer import ThreeLayer as New
from expilict_networks.fourlayer import ThreeLayer as Old
import time

TESTS = 1000

oldAccuracyTotal = 0
oldTrainingTimeTotal = 0

newAccuracyTotal = 0
newTrainingTimeTotal = 0

for _ in range(TESTS):
    st = time.process_time_ns()
    oldAccuracyTotal += nnv1((1, 2, 1), 10).train(TRAIN)
    end = time.process_time_ns()
    oldTrainingTimeTotal += end - st

    st = time.process_time_ns()
    newAccuracyTotal += New().train()
    end = time.process_time_ns()
    newTrainingTimeTotal += end - st


print("\n\n")
print(
    f"After {TESTS} tests old has an avg accuracy of {oldAccuracyTotal/TESTS} and avg trainging time of {oldTrainingTimeTotal/TESTS}s"
)
print(
    f"After {TESTS} tests new has an avg accuracy of {newAccuracyTotal/TESTS} and avg training time of {newTrainingTimeTotal/TESTS}s"
)
