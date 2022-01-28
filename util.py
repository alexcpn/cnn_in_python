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
    """Compute softmax values for each sets of scores in x.
    why the max - see Eli's post https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    """
    e_x = np.exp(x - np.max(x)) # 
    return e_x / e_x.sum()


def derv_softmax(s): # where x is the input
    """
    x is input vector of shape 1* N
    and x is the softmax vector of x with same shape as x
    Derivative of softmax is a Jacobian matrix of size N^2
    we have to take derivative of softmax for each input x = 1 to N
    And since softmax is a vector we need to take derivative of each element in vector  1 tto N
    
    Assuming s.shape == x.shape (3) then the Jacobian (Jik) of the derivative is given below (shape is np.diag(s))

    ds1/dx1 ds1/dx2 ds1/dx3
    ds2/dx1 ds2/dx2 ds2/dx3
    ds3/dx1 ds3/dx2 ds3/dx3

    Note - we dont know x vector; but that's fine; as dsk/dxi= sk(kronecker_ik - si) from (2)
    whe sk is softmax of kth element and si of ith element ( and remember we are working with softmax input)
    Here k goes from 1 to N in outer loop
      and     i goes from 1 to N in inner loop to give us the above Jacobian ik

    (1) https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    (2) https://e2eml.school/softmax.html
    (3) https://stackoverflow.com/a/46028029/429476
    """
    N = len(s)
    Jik =  np.zeros(shape=(N,N))
    for k in range(0, N):
        for i in range(0, N):
            kronecker_ik = 1 if i ==k else 0
            Jik[k][i] = s[k]* (kronecker_ik -s[i]) 
    return Jik




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
    print("------------------------------------------")
    x = np.array([1,2])
    print("x =",x)
    s = softmax(x)
    print("softmax",s)
    Jik = derv_softmax(s)
    print("Derivative of softmax Jacobian is ",Jik)
  
    

