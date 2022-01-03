import unittest

from numpy.lib.type_check import imag
import cnn
import numpy as np


class TestConv2D(unittest.TestCase):

    def test_conv2d(self):
        testConv2D = cnn.Conv2D()
        filter_size = 2       
        weight1 =  np.ones((filter_size,filter_size,3)) 
        image = np.arange(16).reshape(4, 4)
        print("Intial Image=",image)
        image = np.stack([image,image,image], axis=2) # to mimick RGB channel
        print("Image Shape=",image.shape)
        print("Image [0,0,0]=",image[0,0,0],"[1,0,1]=",image[1,0,1],"[2,0,2]=",image[2,0,2])
        print("Image [0,0,:]=",image[0,0,:])
        conv_activation= testConv2D.conv2d(image,weight1)
        print("Final Acitvation Shape=",conv_activation.shape)
        print("Final Acitvation =",conv_activation)




if __name__ == '__main__':
    unittest.main()