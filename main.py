from numpy.core.fromnumeric import shape
import cnn
import util as util
import numpy as np


def relu(z):
    '''Relu activation function'''
    # if num > 0 then num else 0
    output = max(z, 0)
    return(output)
def relu_derivative(z):
    return np.greater(z, 0).astype(int)

if __name__ == '__main__':

    # Generate a random imaage
    image_size = 32 
    image_depth = 3
    image = np.arange(image_size*image_size).reshape(image_size, image_size)
    # to mimick RGB channel
    image = np.stack([image,image,image], axis=image_depth-1) # 0 to 2
    print("Image Shape=",image.shape)
    #print("Image [0,0,:]=",image[0,0,:])

    # The class containing the convolution Logic
    testConv2D = cnn.Conv2D()
    filter_size = 5  
    number_of_filters = 6
    weight_layer1 = []

    # Intialize the weight's/filters of Layer1
    for i in range(number_of_filters):
        weight =  np.random.rand(filter_size,filter_size,image_depth) 
        weight_layer1.append(weight)

    convolution_list_1 = []
    for weight in weight_layer1:
        conv = testConv2D.conv2d(image,weight) 
        convolution_list_1.append(conv) 
    
    # We need to stack the convolutions together
    conv_1_stack  = np.stack(convolution_list_1,axis=2)
    print("Convolution Shape after layer 1=",conv_1_stack.shape)

    # Apply activation to layer 1 output
    input_layer2 = util.ReLU(conv_1_stack)
    print("Acitvation Shape after layer 1=",input_layer2.shape)
