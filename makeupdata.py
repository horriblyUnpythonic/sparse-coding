__author__ = 'scip'

import numpy as np
import matplotlib.pyplot as plt

number_of_data_sets = 100
number_of_atoms = 80
data_length = 30


def make_sin_vec():
    start = -abs(np.random.rand() * np.random.randint(6, 15))
    stop = abs(np.random.rand() * np.random.randint(6, 150))
    x = np.linspace(start, stop, data_length)
    wave = np.sin(x).reshape(data_length, 1)
    return wave * np.random.rand()


representation = np.zeros([number_of_atoms, number_of_data_sets])
for j in range(number_of_data_sets):
    n = np.random.randint(1, number_of_atoms / 2)
    for _ in range(n):
        i = np.random.randint(0, number_of_atoms)
        representation[i, j] = np.random.rand()

columns = []
for _ in range(number_of_atoms):
    columns.append(make_sin_vec())

dictionary = np.hstack(columns)
dictionary /= np.linalg.norm(dictionary)

data = np.dot(dictionary, representation)

if __name__ == '__main__':
    print dictionary.shape
    print data.shape

    plt.plot(data)
    plt.show()
    plt.plot(dictionary)
    plt.show()
