
import numpy as np

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

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
    k = np.array([1,3,2])
    r =np.exp(k)
    sum = np.sum(r,axis=0)
    sfm = r/sum
    print(sfm,np.sum(sfm,axis=0))