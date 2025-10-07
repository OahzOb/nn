import random,copy
import numpy as np
random.seed(42)


class Neuron:
    def __init__(self,weight:list[float,],bias:float):
        self.weight=weight
        self.bias=bias
    def forward(self,x:list[float,]):
        y=0
        for w,x0 in zip(self.weight,x):
            y+=w*x0
            y+=self.bias
            return [y]


class Layer:
    def __init__(self,input_size:int,output_size:int):
        neurons=list()
        for _ in range(output_size):
            weight=[random.uniform(-1.0,1.0) for _ in range(input_size)]
            bias=random.uniform(-1.0,1.0)
            neuron = Neuron(weight=weight,bias=bias)
            neurons.append(neuron)
        self.neurons=neurons
        self.input_size=input_size
        self.output_size=output_size
    def forward(self,x:list[float,]) -> list[float]:
        if len(x)!=self.input_size:
            raise ValueError("Dims not matched!")
        output=list()
        for neuron in self.neurons:
            y=neuron.forward(x)
            output+=y
        return output


class Network:
    def __init__(self):
        self.Layers=list()
        self.Layers.append(Layer(2,64))
        self.Layers.append(Layer(64, 128))
        self.Layers.append(Layer(128, 1024))
        self.Layers.append(Layer(1024, 256))
        self.Layers.append(Layer(256, 16))
    def forward(self,x:list[float,]):
        if len(x)!=self.Layers[0].input_size:
            raise ValueError()
        for layer in self.Layers:
            y=layer.forward(x)
            print(y)
            x=copy.deepcopy(y)
        output=y
        return output
if __name__=="__main__":
    net=Network()
    x=[1]*2
    y=net.forward(x=x)
    print(y)
    print("len of y",len(y))













