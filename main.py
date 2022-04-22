from random import shuffle
from time import time
from typing import List
import numpy as np


class BP:
    weights: List[np.matrix] = []
    lr = 0.3  # Скорость обучения

    def MSE(self, y, t):
        return 0.5 * np.sum((y - t)**2)  # Среднеквадратичная ошибка

    def __sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def __init__(self, layers: List[int]):
        for index in range(0, len(layers) - 1):
            wh = np.random.uniform(
                low=-0.3, high=0.3, size=(layers[index+1], layers[index]))
            wh = np.matrix(wh)
            self.weights.append(wh)

        print(f'Веса: {self.weights} \n')

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
                error = (
                    self.weights[index+1].T * weights_delta.reshape(len(weights_delta), 1)).A1

            if index == 0:
                prev_out = input
            else:
                prev_out = outputs[index - 1]

            gradient = outputs[index] * \
                (1 - outputs[index])  # Производная сигмоиды
            weights_delta = error * gradient
            weights_delta = np.array(weights_delta)
            final_delta = prev_out.reshape(
                len(prev_out), 1) * weights_delta.reshape(1, len(weights_delta)) * self.lr
            self.weights[index] -= final_delta.T

    def report(self, time, epochs, average):
        print("---- Отчет ----")
        print(f"{round(time, 3)} секунд")
        print(f"{epochs} эпох")
        print(f"{average} средняя ошибок MSE")
        print(f"--- Конец отчета ---")


bp = BP([3, 2, 3, 2])

train = [
    ([0, 0, 0], [0, 0]),
    ([0, 0, 1], [1, 1]),
    ([0, 1, 0], [0, 0]),
    ([0, 1, 1], [0, 1]),
    ([1, 0, 0], [1, 0]),
    ([1, 0, 1], [1, 1]),
    ([1, 1, 0], [0, 0]),
    ([1, 1, 1], [0, 1]),
]

start_time = time()
epochCount = 0
average = 0

for i in range(0, 3000):
    shuffle(train)
    for input, correct in train:
        bp.train(input, correct)

    mses: list[float] = []
    for input, correct in train:
        output = bp.predict(input)
        mse = bp.MSE(output, correct)
        mses.append(mse)

    average = sum(mses) / len(mses)
    bp.lr *= 0.9999999
    if average < 0.05:
        epochCount = i
        break
    else:
        epochCount = i + 1


finish = time() - start_time

res = bp.predict([0, 0, 1])
print(res)

bp.report(finish, epochCount, average)
