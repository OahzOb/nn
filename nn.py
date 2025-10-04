import random
import numpy as np
random.seed(42)


class Neuron:
    def __init__(self, weight: list[float, ], bias: float):
        self.weight = weight
        self.bias = bias

    def forward(self, x: list[float, ]) -> list[float]:
        y = 0
        for w, x0 in zip(self.weight, x):
            y += w * x0
        y += self.bias
        return [y]


class Layer:
    def __init__(self, input_size: int, output_size: int):
        neurons = list()
        for _ in range(output_size):
            weight = [random.uniform(-1.0, 1.0) for _ in range(input_size)]
            bias = random.uniform(-1.0, 1.0)
            neuron = Neuron(weight=weight, bias=bias)
            neurons.append(neuron)
        self.neurons = neurons
        self.input_size = input_size
        self.output_size = output_size
    def forward(self, x: list[float, ]) -> list[float, ]:
        if len(x) != self.input_size:
            raise ValueError("Dims not matched!")
        output = list()
        for neuron in self.neurons:
            y = neuron.forward(x)
            output += y # output.append(y[0])
        # ReLU
        output_activated = list()
        for y in output:
            if y > 0:
                output_activated.append(y)
            else:
                output_activated.append(0)
        # Sigmoid
        output_activated = list()
        for y in output:
            y_new = 1 / (1 + np.exp(-y))
            output_activated.append(float(y_new))
        return output_activated


if __name__ == '__main__':
    layer = Layer(input_size=10, output_size=20)
    x = [1] * 10
    y = layer.forward(x=x)
    print(y)
    print('len of y:', len(y))
