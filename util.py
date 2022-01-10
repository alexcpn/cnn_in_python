""" Utility functions
"""
import numpy as np

__author__ = "Alex Punnen"
__status__ = "Test"

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

# let us code our sigmoid funciton
def sigmoid(x):
    return 1/(1+np.exp(-x))

# let us add a method that takes the derivative of x as well
def derv_sigmoid(x):
   return sigmoid(x)*(1-sigmoid(x))

# softmax numercially stable

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



if __name__ == '__main__':
    k = np.random.randint(-2,5, size=(2, 4))
    print(k)
    print(ReLU(k))
    print(dReLU(k))
    x = np.array([1828702112.88,1898656673.75, 1822398758.45, 1473265004.06, 1531601321.60,\
        1611254519.90, 1486370895.63, 1849909951.60, 1970530649.00, 1775162782.03])
    #r =np.exp(k)

    v_min = x.min(axis=(0), keepdims=True)
    v_max = x.max(axis=(0), keepdims=True)
    x =(x - v_min)/(v_max - v_min)
    print("x normalized",x)
    r =np.exp(x - np.max(x))
    sum = np.sum(r,axis=0)
    sfm = r/sum
    print(sfm,np.sum(sfm,axis=0))
    print("sfm",softmax(x))