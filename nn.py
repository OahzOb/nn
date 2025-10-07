import random
import numpy as np
import copy
import matplotlib.pyplot as plt
# random.seed(42)


def transpose(matrix: list[list, ]) -> list[list, ]:
    matrix_new = list()
    n = len(matrix)
    m = len(matrix[0])
    for i in range(m):
        row = list()
        for j in range(n):
            row.append(matrix[j][i])
        matrix_new.append(row)
    return matrix_new


def sigmoid(y: float) -> float:
    y_new = 1 / (1 + np.exp(-y))
    return float(y_new)


class Neuron:
    def __init__(self, weight: list[float, ], bias: float):
        self.weight = weight
        self.bias = bias
        self.input = list()
    def forward(self, x: list[float, ]) -> list[float]:
        self.input = copy.deepcopy(x)
        y = 0
        for w, x0 in zip(self.weight, x):
            y += w * x0
        y += self.bias
        return [y]
    def backward(self, grad: float, lr: float = 1e-6) -> list[float, ]:
        weight_new = list()
        weight_origin = copy.deepcopy(self.weight)
        for x, w in zip(self.input, self.weight):
            weight_new.append(w - lr * grad * x)
        self.weight = copy.deepcopy(weight_new)
        self.bias -= lr * grad
        return weight_origin

class Layer:
    def __init__(self, input_size: int, output_size: int, activation: str = 'Sigmoid'):
        neurons = list()
        for _ in range(output_size):
            weight = [random.uniform(-1.0, 1.0) for _ in range(input_size)]
            bias = random.uniform(-1.0, 1.0)
            neuron = Neuron(weight=weight, bias=bias)
            neurons.append(neuron)
        self.neurons = neurons
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
    def forward(self, x: list[float, ]) -> list[float, ]:
        if len(x) != self.input_size:
            raise ValueError("Dims not matched!")
        output = list()
        for neuron in self.neurons:
            y = neuron.forward(x)
            output += y # output.append(y[0])
        self.output = copy.deepcopy(output)
        if self.activation == 'ReLU':
            output_activated = list()
            for y in output:
                if y > 0:
                    output_activated.append(y)
                else:
                    output_activated.append(0)
        elif self.activation == 'Sigmoid':
            output_activated = list()
            for y in output:
                output_activated.append(sigmoid(y))
        else:
            output_activated = output
        return output_activated
    def backward(self, grads: list[float, ], lr: float = 1e-6) -> list[float, ]:
        dzdas = list()
        dadzs = list()
        if self.activation == 'ReLU':
            for y in self.output:
                if y > 0:
                    dadzs.append(1.0)
                else:
                    dadzs.append(0.0)
        elif self.activation == 'Sigmoid':
            for y in self.output:
                dadzs.append(sigmoid(y) * (1 - sigmoid(y)))
        else:
            dadzs = [1.0] * len(self.output)
        dLdzs = [grad * dadz for grad, dadz in zip(grads, dadzs)]
        for neuron, dLdz in zip(self.neurons, dLdzs):
            dzda = neuron.backward(grad=dLdz, lr=lr)
            dzdas.append(dzda)
        dzdas = transpose(dzdas)
        dzdas_new = list()
        for dzda in dzdas:
            ds = list()
            for d, dLdz in zip(dzda, dLdzs):
                ds.append(d * dLdz)
            dzdas_new.append(sum(ds))
        return dzdas_new


class Network:
    def __init__(self):
        self.Layers = list()
        # self.Layers.append(Layer(input_size=1, output_size=64))
        # self.Layers.append(Layer(input_size=64, output_size=128))
        # self.Layers.append(Layer(input_size=128, output_size=1024, activation='ReLU'))
        # self.Layers.append(Layer(input_size=1024, output_size=128, activation='ReLU'))
        # self.Layers.append(Layer(input_size=128, output_size=1, activation='none'))

        self.Layers.append(Layer(input_size=1, output_size=16, activation='ReLU'))
        self.Layers.append(Layer(input_size=16, output_size=128, activation='ReLU'))
        self.Layers.append(Layer(input_size=128, output_size=16, activation='ReLU'))
        self.Layers.append(Layer(input_size=16, output_size=1, activation='none'))
        self.output = list()
    def forward(self, x: list[float, ]) -> list[float, ]:
        if len(x) != self.Layers[0].input_size:
            raise ValueError("Dims not matched!")
        for layer in self.Layers:
            y = layer.forward(x)
            # print(y)
            x = copy.deepcopy(y)
        output = y
        self.output = copy.deepcopy(output)
        return output
    def backward(self, y_true: list[float, ], lr: float = 1e-6) -> None:
        dLdas = [yp - yt for yp, yt in zip(self.output, y_true)]
        for layer in self.Layers[::-1]:
            dLdas = layer.backward(grads=dLdas, lr=lr)


def loss_func(y_pred: list[float, ], y_true: list[float, ]) -> float:
    losses = [(yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)]
    loss = sum(losses) / len(y_pred)
    return loss


if __name__ == '__main__':
    net = Network()
    epochs = 10000
    x = np.linspace(-np.pi, np.pi, 1000)
    y = np.sin(x)
    x = x.tolist()
    y = y.tolist()
    indices = list(range(len(x)))
    for epoch in range(epochs):
        idx = random.choice(indices)
        x0 = [x[idx]]
        y0 = [y[idx]]
        y0_pred = net.forward(x=x0)
        loss = loss_func(y_pred=y0_pred, y_true=y0)
        print(f'epoch: {epoch+1}/{epochs} | loss:', loss)
        net.backward(y_true=y, lr=1e-5)
    y_pred = list()
    for x0 in x:
        y_pred.append(net.forward(x=[x0]))
    plt.plot(x, y, label='True')
    plt.plot(x, y_pred, label='Prediction')
    plt.legend()
    plt.show()
