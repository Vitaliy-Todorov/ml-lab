import math
import numpy as np
import random

class Neuron:
    def __init__(self):
        self._parameters = np.zeros(3)
        self._learning_rate = 0.1

    def target_label(self, a, b):
        if a >= b:
            return 1
        else:
            return 0

    def predict_probability(self, a, b):
        z = self._parameters[0] * a + self._parameters[1] * b + self._parameters[2]
        return self._activation_function(z)
    
    def train(self):
        for _ in range(10):
            a = random.uniform(-100, 100)  
            b = random.uniform(-100, 100)  
            self._train_step(a, b)
            
        
    def _activation_function(self, z):
        return 1 / (1 + math.exp(-z))
    
    def _train_step(self, a, b):
        y = self.target_label(a, b)
        p = self.predict_probability(a, b)

        grad = np.zeros(3)

        grad[0] = (p - y) * a # dL/dw1
        grad[1] = (p - y) * b # dL/dw2
        grad[2] = p - y # dL/dc
        
        self._parameters = self._parameters - self._learning_rate * grad