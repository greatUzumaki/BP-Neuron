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



# test = np.matrix(([1, 3, 5], [2, 4, 6]))
# res = np.dot(test, [0, 1, 1])
# print(res)


# class BP:
#     weights = []
#     lr = 0.3

#     def __init__(self, configArray) -> None:
#         for config in range(0, len(configArray) - 1):
#             wh = np.random.uniform(
#                 low=-0.3, high=0.3, size=(configArray[config+1], configArray[config]))
#             self.weights.append(wh)

#     def sigmoid(self, x):
#         return 1/(1 + np.exp(-x))

#     def derivatives_sigmoid(self, x):
#         return x * (1 - x)

#     def predict(self, input_vector):
#         neurons_sums = []

#         for layer in range(0, len(self.weights)):
#             sum = []

#             for weight in range(0, len(self.weights[layer])):
#                 scalar = np.dot(input_vector, self.weights[layer][weight])
#                 sum.append(self.sigmoid(scalar))

#             neurons_sums.append(sum)
#             input_vector = sum

#         return neurons_sums

#     def backprop(self, neuron_sums, X, Y):
#         global_error = Y - neuron_sums[-1]
#         errors = []

#         for error in range(0, len(global_error)):
#             global_error *= self.derivatives_sigmoid(neuron_sums[-1][error])

#         for layer in reversed(range(1, len(self.weights))):
#             error_sum = []

#             for weight in range(0, len(self.weights[layer])):
#                 delta = np.dot(
#                     global_error, self.weights[layer]) * self.derivatives_sigmoid(neuron_sums[layer][weight])
#                 error_sum.append(self.sigmoid(delta))

#             errors.append(error_sum)

#         return errors


# bp = BP([3, 2, 1])

# X = np.array(([3, 9, 3], [2, 5, 2], [3, 6, 2]), dtype=float)
# Y = np.array(([1], [0], [1]), dtype=float)

# for epoch in range(0, 1):
#     for data in range(0, len(X)):
#         result = bp.predict(X[data])
#         bp.backprop(result, X[data], Y[data])
#         bp.update()
