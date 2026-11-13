import math
import numpy as np
import random

class Neuron:
    def __init__(self):
        self._parameters = np.zeros(3)

    def target_label(self, a, b):
        if a >= b:
            return 1
        else:
            return 0

    def predict_probability(self, a, b):
        z = self._parameters[0] * a + self._parameters[1] * b + self._parameters[2]
        return self._activation_function(z)

    def train(self, training_steps=3, batch_size=10, learning_rate=0.1):
        for _ in range(training_steps):
            self._train_step(batch_size, learning_rate)

    def _train_step(self, batch_size, learning_rate):
        gradient_sum = np.zeros(3)

        for _ in range(batch_size):
            a = random.uniform(-100, 100)
            b = random.uniform(-100, 100)
            gradient_sum += self._get_gradient(a, b)

        gradient_sum = gradient_sum / batch_size
        self._parameters = self._parameters - learning_rate * gradient_sum

    def _activation_function(self, z):
        return 1 / (1 + math.exp(-z))

    def _get_gradient(self, a, b):
        y = self.target_label(a, b)
        p = self.predict_probability(a, b)

        grad = np.zeros(3)

        grad[0] = (p - y) * a  # dL/dw1
        grad[1] = (p - y) * b  # dL/dw2
        grad[2] = p - y  # dL/dc

        return grad
