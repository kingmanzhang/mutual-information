import numpy as np


def say_hello(name):
    if name is None:
        return 'Hello, world!'
    else:
        return f'Hello, {name}!'


def square_matrix(n):
    return np.array(np.random.random(size=n*n)).reshape((n, n))
