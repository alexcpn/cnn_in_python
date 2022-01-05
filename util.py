
import numpy as np

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)


if __name__ == '__main__':
    k = np.random.randint(-2,5, size=(2, 4))
    print(k)
    print(ReLU(k))
    print(dReLU(k))