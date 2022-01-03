from numpy.core.fromnumeric import shape
import cnn
import numpy as np


if __name__ == '__main__':

    testConv2D = cnn.Conv2D()
    filter_size = 4  
    image_size = 16 
    image_depth = 3    
    
    # Weight's or filters of Layer1
    weight1 =  np.random.rand(filter_size,filter_size,image_depth) 
    weight2 =  np.random.rand(filter_size,filter_size,image_depth)

    # Generate a random imaage
    image = np.arange(image_size*image_size).reshape(image_size, image_size)
    # to mimick RGB channel
    image = np.stack([image,image,image], axis=image_depth-1) # 0 to 2
    print("Image Shape=",image.shape)
    #print("Image [0,0,:]=",image[0,0,:])
    
    # Convolve weight matrxit
    activation_1_1 = testConv2D.conv2d(image,weight1)
    activation_1_2 = testConv2D.conv2d(image,weight2)
    print("Acitvation Shape=",activation_1_1.shape)
    
    #We need to layer the activations into a new image

    activation_1  = np.stack([activation_1_1,activation_1_2],axis=2)
    print("Acitvation Shape after layer 1=",activation_1.shape)
