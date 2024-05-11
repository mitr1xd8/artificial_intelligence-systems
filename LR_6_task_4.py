import numpy as np
import neurolab as nl

target = [[1, 0, 0, 0, 1,
           1, 1, 0, 0, 1,
           1, 0, 1, 0, 1,
           1, 0, 0, 1, 1,
           1, 0, 0, 0, 1],
          [1, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 1, 1, 1, 1,
           1, 0, 0, 0, 0,
           1, 1, 1, 1, 1],
          [1, 1, 1, 1, 0,
           1, 0, 0, 0, 1,
           1, 1, 1, 1, 0,
           1, 0, 0, 1, 0,
           1, 0, 0, 0, 1],
          [0, 1, 1, 1, 0,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           1, 0, 0, 0, 1,
           0, 1, 1, 1, 0]]

chars = ['N', 'E', 'R', 'O']
target = np.asfarray(target)
target[target == 0] = -1

net = nl.net.newhop(target)

output = net.sim(target)
print("Test on train samples:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())

print("\nTest on defaced N:")
test = np.asfarray([0, 0, 0, 0, 0,
                    1, 1, 0, 0, 1,
                    1, 1, 0, 0, 1,
                    1, 0, 1, 1, 1,
                    0, 0, 0, 1, 1])
test[test == 0] = -1
out = net.sim([test])
print((out[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))
print("\nTest on defaced N:")
test = np.asfarray([0, 0, 0, 0, 0,
                    1, 1, 0, 0, 1,
                    1, 1, 0, 0, 1,
                    1, 0, 1, 1, 1,
                    0, 0, 0, 1, 1])
test[test == 0] = -1
out = net.sim([test])
print((out[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))
print("\nTest of defaced E:")
test = np.asfarray([1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0,
                    1, 1, 0, 1, 1,
                    1, 0, 0, 0, 0,
                    1, 0, 1, 1, 1])
test[test == 0] = -1
output = net.sim([test])
print((output[0] == target[1]).all(), 'Sim. steps', len(net.layers[0].outs))
print("\nTest of defaced R:")
test = np.asfarray([1, 1, 0, 1, 0,
                    1, 0, 0, 0, 1,
                    1, 1, 1, 1, 0,
                    0, 0, 0, 1, 0,
                    1, 0, 0, 0, 1])
test[test == 0] = -1
output = net.sim([test])
print((output[0] == target[2]).all(), 'Sim. steps', len(net.layers[0].outs))
print("\nTest of defaced O:")
test = np.asfarray([0, 1, 1, 1, 0,
                    1, 0, 0, 0, 1,
                    0, 0, 1, 0, 1,
                    1, 0, 0, 0, 1,
                    0, 1, 0, 1, 0])
test[test == 0] = -1
output = net.sim([test])
print((output[0] == target[3]).all(), 'Sim. steps', len(net.layers[0].outs))
