# Issues

This file is for issue that are unsolve and have multiple solutions that still need to be considerd

## Activation funtion problem:

### Activation and weights between two iterations:

[7]
| 0.500 |
| 5.363 |
| 5.312 |
| 5.312 |
| 0.500 |
[3.5, 37.53909030841787, 37.185760927939974, 37.185760927939974, 3.5]
| 255.827 263.827 260.827 260.827 263.827 |
[31120.72716345309]

[8]
| -1.250 |
| -13.407 |
| -13.281 |
| -13.281 |
| -1.250 |
[0, 0, 0, 0, 0]
| -855.127 -847.127 -850.127 -850.127 -847.127 |
[0.0]

### Why is this happening

all activations are negative so relu funtion is messing it up
why doesnt the network fix this in futur training iterations
these weights remain untouched
it is because when the "better layer2" is calculated, the relu funtion is apliead and because weights23 are
all negative too so the cost of layer2 to betterlayer2 is 0 so no change is aplied

### Solutions

Switch to leaky relu funtion. Tested works far better.
Switch to sigmoid funtion
Add training rate multiplier

### Testing results

Not tested yet

## Overshooting issue

### Activation and weights between two iterations:

[1]
| 7.000 |
| 2.000 |
| -2.000 |
| -10.000 |
| 4.000 |
[7, 2, 0, 0, 4]
| 2.000 6.000 7.000 -8.000 -1.000 |
[22]

[2]
| 7.000 |
| 2.000 |
| -2.000 |
| -10.000 |
| 4.000 |
[14, 4, 0, 0, 8]
| -3.000 1.000 2.000 -13.000 -6.000 |
[-86.0]

[3]
| 7.000 |
| 2.000 |
| -2.000 |
| -10.000 |
| 4.000 |
[21, 6, 0, 0, 12]
| 8.250 12.250 13.250 -1.750 5.250 |
[309.75]

### Why is this happening

All waits are changed the same amout based on the cost
i.e if the cost from a layer with one node to another with one node is 10 the weight will be changed x amount.
However if that first layer had 100 nodes, each weight will still be changed x amount, which is 100 times more than each weight should have been changed to achieve the same result as the first example

### Solutions

Adjust each weights by dividng the adjustment by the amount of nodes

Adjusting each weight based on the signifigance of the weight i.e how much the weight affected the nodes value
If so, put emphasis on bigger weight or smaller weights?

### Testing results

Not tested yet
