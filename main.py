from typing import List
import numpy as np


class BP:
    weights = []
    lr = 0.3

    def __sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def __init__(self, layers: List[int]):
        for index in range(0, len(layers) - 1):
            wh = np.random.uniform(
                low=-0.3, high=0.3, size=(layers[index+1], layers[index]))
            wh = np.matrix(wh)
            self.weights.append(wh)

        print(f'Веса: {self.weights}')

    def __predict(self, vector):
        outputs: List[np.ndarray] = []  # Ответ каждого слоя
        input = vector

        for matrix in self.weights:
            input = np.dot(matrix, input)
            input = input.A1
            input = list(map(self.__sigmoid, input))
            input = np.array(input)
            outputs.append(input)

        return outputs

    def predict(self, vector):  # Предикт для людей
        return self.__predict(vector)[-1]

    def train(self, input, correct):
        correct = np.array(correct)
        input: np.ndarray = np.array(input)

        outputs = self.__predict(input)  # Получение массива ответов
        error = []  # Ошибка конкретных нейронов
        weights_delta = []  # Смещение весов
        prev_out = []

        for index in reversed(range(0, len(self.weights))):
            if index == len(self.weights) - 1:
                error = outputs[index] - correct
            else:
                error = (self.weights[index + 1].T * weights_delta).A1

            if index == 0:
                prev_out = input
            else:
                prev_out = outputs[index - 1]

            gradient = outputs[index] * (1 - outputs[index])
            weights_delta = error * gradient
            final_delta = prev_out.reshape(
                len(prev_out), 1) * weights_delta.reshape(1, len(weights_delta)) * self.lr
            self.weights[index] -= final_delta.T


bp = BP([3, 2, 1])
