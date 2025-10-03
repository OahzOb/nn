import random
import numpy as np
random.seed(42)


class Neuron:
    def __init__(self, weight: float, bias: float):
        self.weight = weight
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = self.weight @ x + self.bias
        return y


class Layer:
    def __init__(self, input_size: int, output_size):
        neurons = list()
        for _ in range(num_neuron):
            weight = random.random()
            bias = random.random()
            neuron = Neuron(weight=weight, bias=bias)
            neurons.append(neuron)
        self.neurons = neurons

    def forward(self, x: float) -> :


if __name__ == '__main__':
    n1 = Neuron(1, 1)
    print(n1.forward(10))
