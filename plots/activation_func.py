import matplotlib.pyplot as plt
import numpy as np
import math
import copy


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relu(x):
    return max([0, x])


# init tables
X = np.arange(start=-5, stop=5, step=0.05)
Y = []

for j in range(3):
    Y.append(copy.copy(X))

# calculate values for activation functions
for i in range(len(X)):
    Y[0][i] = sigmoid(Y[0][i])
    Y[1][i] = relu(Y[1][i])
    Y[2][i] = math.tanh(Y[2][i])

# plot the activation functions separately
X_ticks = np.arange(-5, 6)
fig, ax = plt.subplots(1, 3)
fig.set_figheight(3.7)
fig.set_figwidth(20)

titles = ["Sigmoid", "Hyperbolic Tangent", "Rectified Linear Unit"]

for idx, y in enumerate(Y):
    ax[idx].plot(X, y)
    ax[idx].set_title(titles[idx])
    ax[idx].grid(b=True, ls='-.')
    ax[idx].set_xticks(X_ticks)
    ax[idx].set_xlabel("input")
    ax[idx].set_ylabel("output")

plt.show()

# plot the activation functions together
colors = ['#990000', '#000000', '#2F3EEA']

for idx, y in enumerate(Y):
    plt.plot(X, y, c=colors[idx])

plt.grid(ls='-.')
plt.xticks(X_ticks)
plt.xlabel("input")
plt.ylabel("output")
plt.legend(loc='right')
plt.show()

