"""
 The module that does the Convolution Operation
"""
import numpy as np
from numpy.lib.type_check import imag

__author__ = "Alex Punnen"
__status__ = "Test"
class Conv2D:
    def __init__(self):
        pass

    def conv2d(self, image, filter):
        imagesize = image.shape
        filter_size = filter.shape
        #print(imagesize)
        i = imagesize[0]
        j = imagesize[1]
        k = imagesize[2]
        fi = filter_size[0]  # filter length of row
        fj = filter_size[1] 
        fk = filter_size[2] 

        #print("Image (i,j,k)=", i, j, k, "Filter (i,j,k)=", fi,fj,fk)
        di = -1
        # Out=W−F+1 https://stats.stackexchange.com/a/323423/191675
        convolution_size = i-fi+1
        interim_ouput = np.zeros((convolution_size, convolution_size))

        for m in range(0, i):  # x axis
            di += 1
            dj = -1
            for n in range(0, j):  # y axis
                dj += 1
                if (m+fi) <= i and (fi+n) <= j:
                    # print(m,':',n,':',o,"\n------------\n",image[m:fl+m,n:fl+n].shape)
                    # temp = np.dot(image[m:fl+m,n:fl+n,o],filter) # this is wrong - we 
                    # need to unravel the selected box part first, to do dot product with weights
                    #  https://stats.stackexchange.com/a/335500/191675
                    temp = np.dot(
                        image[m:fi+m, n:fi+n].ravel(), filter.ravel())
                    interim_ouput[di, dj] = temp
                    #print("Inner Ouput \n interim_ouput[{},{},{}]={}\n".format(
                    #    di, dj, dk, interim_ouput[di, dj, dk]))
                #print("Intermin Ouput \n {}\n".format(interim_ouput))
        #print("Ouput of Convolution=", interim_ouput.shape)
        return interim_ouput
