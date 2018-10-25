import math
import itertools
import random

random.seed(13)

size(1600,800)
background(255)

def sigmoid(value):
    return 1 / (1 + math.exp(-value))


class SigmoidNeuron:
    def __init__(self, weights, bias, x, y):
        self.weights = weights
        self.bias = bias
        self.x, self.y = x,y
        

    def __repr__(self):
        return "SigmoidNeuron(" + str(self.weights) + " " + str(self.bias)

    def feed_forward(self, inputs):
        assert len(inputs) == len(self.weights)
        scaled_inputs = [weight * input_ for weight, input_ in zip(self.weights, inputs)]
        summed_result = sum(scaled_inputs)
        result = summed_result + self.bias
        
        if self.weights == [1,1,1,1]:
            global in_iter
            out = nInput[in_iter]
            in_iter+=1
            return out
        return sigmoid(result)
    
    def display(self):
        strokeWeight(2)
        fill(255)
        ellipse(self.x, self.y, 60, 60)
        fill(0)
        text("Weights: " + str([round(w,5) for w in self.weights]), self.x - 30,self.y - 50)
        text("Bias: " + str(self.bias), self.x - 30,self.y - 35)

class NeuralNetwork:
    def __init__(self,layers):
        self.layers = layers
    def feed_forward(self, input_):
        neurons = []
        for layer in self.layers:
            if neurons:
                for i in range(len(neurons)):
                    for j in range(len(layer)):
                        weight = layer[j].weights[i]
                        strokeWeight((weight**2)*5)
                        line(layer[j].x ,layer[j].y ,neurons[i].x ,neurons[i].y)
                
            neurons = []            
            new_input = []
            for n in layer:
                result = n.feed_forward(input_)
                new_input.append(result)
                neurons.append(n)
                n.display()
                textSize(10)
                text("Output:\n  " + str(round(result, 3)), n.x - 20, n.y - 3)
                print(n, result)
            input_ = new_input
            print("\n")
        return input_

def make_neuron(inputs_size, x_loc, y_num):
    fill(255)
    strokeWeight(3)
    global y_iter
    y_loc = height/(y_num + 1) * (y_iter)
    y_iter += 1
    
    global n_iter
    neur = neur_list[n_iter]
    n = SigmoidNeuron(neur[0], neur[1], x_loc, y_loc)
    n_iter += 1
    return n

    

global y_iter
y_iter = 1
global n_iter
n_iter = 0
global in_iter
in_iter = 0

nInput = [0.005, 0.2, 0.8, 0.2]

neur_list = [
[[1,1,1,1],0],[[1,1,1,1],0],[[1,1,1,1],0],[[1,1,1,1],0],
[[-0.05085864262290385, 0.31494500531451064, 0.3328209422496762, -0.7147992941492645], -1.967418670729796] ,
[[-0.25049101587327116, -0.4519037210433372, 0.6206961044701675, 0.3811853060773691], -0.19562888381680055] ,
[[0.11638009550669604, 0.322642171744002, -0.709394410579794, -0.11989041615046037], -1.513198032833334] ,
[[0.8119459630586452, -0.8823513451782679, 0.6376402556841607, -0.8507807327282451], 0.060836512280625143] ,
[[-0.3260008144164648, -0.19077143374270178, 0.6848067356251235, -0.9627913993759944], -1.8176459114330934] ,
[[0.8300672868198937, 0.017851832855402927, -0.8180440218828333, 0.9742679808111392, 0.8934260374865413], -1.6624170915589207] ,
[[-0.15358677549181277, -0.7298609981016719, -0.3749186116154637, 0.24290389654286448, -0.6729517433620373], 0.09050346655193664] ,
[[-0.8972933656699904, -0.6575718148462824, 0.6313952965081153, -0.19896260488869255, -0.16231957572476685], -0.21208706938646493] ,
[[-0.046463510255569274, -0.2308372178502507, -0.9390112364183674], 0.1789822820424405] ,
[[0.9349177657345891, 0.9526075391069868, 0.32719208689874013], -0.9309644265229744] ,
]

x_offset = 500
y_iter = 1
input_layer = [make_neuron(4, -200 + x_offset, 4) for _ in range(4)]
y_iter = 1
hidden_layer_1 = [make_neuron(4, 100 + x_offset, 5) for _ in range(5)]
y_iter = 1
hidden_layer_2 = [make_neuron(5, 400 + x_offset, 3) for _ in range(3)]
y_iter = 1
output_layer = [make_neuron(3, 700 + x_offset, 2) for _ in range(2)]

nn = NeuralNetwork([input_layer, hidden_layer_1, hidden_layer_2, output_layer])
outputs = nn.feed_forward([0.005, 0.2, 0.8, 0.2])
print(outputs)
