from csv import reader
from random import shuffle
from time import time
from typing import List
import numpy as np


class BP:
    weights: List[np.matrix] = []
    lr = 0.3  # Скорость обучения

    def MSE(self, y, t):
        return 0.5 * np.sum((y - t)**2)  # Среднеквадратичная ошибка

    def NormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def __init__(self, layers: List[int]):
        for index in range(0, len(layers) - 1):
            wh = np.random.uniform(
                low=-0.3, high=0.3, size=(layers[index+1], layers[index]))
            wh = np.matrix(wh, dtype=np.float32)
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

    def load_csv(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(list(map(np.float32, row)))
        return np.array(dataset)

    def report(self, time, epochs, average, pred_aver):
        print("---- Отчет ----")
        print(f"{round(time, 3)} секунд")
        print(f"{epochs} эпох")
        print(f"{round(average, 3)} средняя ошибок MSE на обучающей")
        print(f"{round(pred_aver,8)} на тестовой")
        print(f"--- Конец отчета ---")


bp = BP([4, 5, 3, 1])
threshold = 0.03

filename = 'dataset3.csv'

DATA = bp.load_csv(filename).T
TRAIN = DATA[:-1, :]
CORRECT = np.array(list(map(int, DATA[-1])))

for column in range(len(TRAIN)):
    TRAIN[column] = bp.NormalizeData(TRAIN[column])

TRAIN = TRAIN.T
dataset = list()

for row in range(len(TRAIN)):
    one_row = tuple((list(TRAIN[row]), [CORRECT[row]]))
    dataset.append(one_row)

start_time = time()
epochCount = 0
average = 0
errorCount = 0

# Обучение
for i in range(0, 400):
    shuffle(dataset)
    for input, correct in dataset:
        bp.train(input, correct)

    mses: list[float] = []
    for input, correct in dataset:
        output = bp.predict(input)
        mse = bp.MSE(output, correct)
        if mse > 0.3:
            errorCount += 1
        mses.append(mse)

    average = sum(mses) / len(mses)
    bp.lr *= 0.9999
    if average < threshold:
        epochCount = i
        break
    else:
        epochCount = i + 1


finish = time() - start_time

# Предсказывание
predict_file = 'predict.csv'

PREDICT_DATA = bp.load_csv(predict_file).T
PREDICT = PREDICT_DATA[:-1, :]
CORRECT_PREDICT = np.array(list(map(int, PREDICT_DATA[-1])))
PREDICT = PREDICT.T

predict_dataset = list()
for row in range(len(PREDICT)):
    one_row = tuple((list(PREDICT[row]), [CORRECT_PREDICT[row]]))
    predict_dataset.append(one_row)

predict_mse = []
for input, correct in predict_dataset:
    res = bp.predict(input)
    pred_mse = bp.MSE(np.array(res), np.array(correct))
    predict_mse.append(pred_mse)
    print(res)

predict_average = sum(predict_mse) / len(predict_mse)

bp.report(finish, epochCount, average, predict_average)
