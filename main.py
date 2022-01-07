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

def intializeWeights(number_of_filters,filter_size,depth):
    """Intialize the weight's/filters of Layer
    :return: list of weights
    """
    weight_layer = []
    for i in range(number_of_filters):
        weight =  np.random.rand(filter_size,filter_size,depth) 
        weight_layer.append(weight)
    return weight_layer

def layerConvolutionActivation(image, filter_size,number_of_filters,weight_layer1):
    """This function intializes the random weights as per the specified filter
    size and number and does the Convolution with the filter and then applies the
    activation function to the convolution 
    :return: the output layer after applyigng the Activation
    """
    convolution_list_1 = []
    for weight in weight_layer1:
        conv = testConv2D.conv2d(image,weight) 
        convolution_list_1.append(conv) 
    
    # We need to stack the convolutions together
    conv_1_stack  = np.stack(convolution_list_1,axis=2)
    print("Convolution Shape after layer 1=",conv_1_stack.shape)

    # Apply activation to layer 1 output
    output_layer2 = util.ReLU(conv_1_stack)
    print("Acitvation Shape after layer 1=",output_layer2.shape)
    return output_layer2

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

    # we will create leNet without the Pooling parts
    # (stride is always 1 for now)
    #  [32.32.3] *(5.5.3)*6  == [28.28.6] * (5.5.6)*1 = [24.24.1] *  (5.5.3)*16 = [20.20.16] * FC layer 
    
    # For layer 1
    filter_size = 5  
    number_of_filters = 6
    # Intialize the weight's/filters of Layer1
    weight_layer1 = intializeWeights(number_of_filters,filter_size,image.shape[2])
    output_layer1 = layerConvolutionActivation(image, filter_size,number_of_filters,weight_layer1)

     # For layer 2
    filter_size = 5  
    number_of_filters = 1
    # Intialize the weight's/filters of Layer1
    weight_layer2 =  intializeWeights(number_of_filters,filter_size,output_layer1.shape[2])
    # Do convolution and activation
    output_layer2 = layerConvolutionActivation(output_layer1, filter_size,number_of_filters,weight_layer2)

    # For layer 3
    # Out=W−F+1 imagesize - filtersize + 1
    filter_size = 5  
    number_of_filters = 16
    # Intialize the weight's/filters of Layer1
    weight_layer3 =  intializeWeights(number_of_filters,filter_size,output_layer2.shape[2])
    # Do convolution and activation
    output_layer3 = layerConvolutionActivation(output_layer2, filter_size,number_of_filters,weight_layer3)
    print("output_layer3shape =", output_layer3.shape) # output_layer3 shape = (20, 20, 16)
        
    # Lets add the fully connected layer say 120 - we need the shape to be compatible - for that we are adding the
    # the dimension of the above layer 
    # Note that I don't want to flatten here!

    weight_layer4 =  np.random.rand(output_layer3.shape[0],120,output_layer3.shape[2]) 
    print("Fully Connected Weight4 shape =", weight_layer4.shape) # (20, 120, 16)
    # this time there is no convolution - rather we need to do a dot
    output_layer4 = np.einsum('ijp,jkp->ik', output_layer3, weight_layer4) # (20, 120)
    #output_layer4 = np.tensordot(output_layer3,weight_layer4,axes=2)
    output_layer4 = util.ReLU(output_layer4)
    print("Fully Connected Layer 1 Ouput shape =", output_layer4.shape)

    weight_layer5 =  np.random.rand(output_layer4.shape[1],1) 
    print("Fully Connected Weight5 shape =", weight_layer5.shape) # (20, 120, 16)
    # this time there is no convolution - rather we need to do a dot
    output_layer5 = np.einsum('ij,jk->ik', output_layer4, weight_layer5) # (20, 120)
    output_layer5 = util.ReLU(output_layer5)
    print("Fully Connected Layer 2 Ouput shape =", output_layer5.shape)

    # final layer lets make it 10 classes
    weight_layer6 =  np.random.rand(output_layer5.shape[0],10) 
    print("Fully Connected Weight6 shape =", weight_layer6.shape) # (20, 120, 16)
    # this time there is no convolution - rather we need to do a dot
    output_layer6 = np.einsum('ij,ik->jk', output_layer5, weight_layer6) #  20,1*20,10 (1, 10)
    print("Final  Ouput shape =", output_layer6.shape)
    # Run softmax
    print("Final  Ouput  =", util.softmax(output_layer6))

