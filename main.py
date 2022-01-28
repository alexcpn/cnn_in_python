"""Main module
"""
from numpy.core.fromnumeric import shape
import cnn
import util as util
import numpy as np

__author__ = "Alex Punnen"
__status__ = "Test"

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
    """This function  does the Convolution with the filter and then applies the
    activation function to the convolution 
    :return: the output layer after applyigng the Activation
    """
    convolution_list_1 = []
    for weight in weight_layer1:
        conv = testConv2D.conv2d(image,weight) 
        convolution_list_1.append(conv) 
    
    # We need to stack the convolutions together
    conv_1_stack  = np.stack(convolution_list_1,axis=2)
    print("Convolution Shape =",conv_1_stack.shape)

    # Apply activation to layer 1 output
    output_layer2 = util.ReLU(conv_1_stack)
    #print("Acitvation Shape after layer 1=",output_layer2.shape)
    return output_layer2

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    # Generate a random imaage
    image_size = 32 
    image_depth = 3
    image = np.random.rand(image_size, image_size)
    # to mimick RGB channel
    image = np.stack([image,image,image], axis=image_depth-1) # 0 to 2
    print("Image Shape=",image.shape)
    print("Image [0,0,:]=",image[0,1,2])

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
    #print("val output_layer1= ",output_layer1[0,0,0])

     # For layer 2
    filter_size = 5  
    number_of_filters = 1
    # Intialize the weight's/filters of Layer1
    weight_layer2 =  intializeWeights(number_of_filters,filter_size,output_layer1.shape[2])
    # Do convolution and activation
    output_layer2 = layerConvolutionActivation(output_layer1, filter_size,number_of_filters,weight_layer2)
    #print("val output_layer2= ",output_layer2[0,0,0])

    # For layer 3
    # Out=W−F+1 imagesize - filtersize + 1
    filter_size = 5  
    number_of_filters = 16
    # Intialize the weight's/filters of Layer1
    weight_layer3 =  intializeWeights(number_of_filters,filter_size,output_layer2.shape[2])
    # Do convolution and activation
    output_layer3 = layerConvolutionActivation(output_layer2, filter_size,number_of_filters,weight_layer3)
    print("Output_layer 3 shape =", output_layer3.shape) # output_layer3 shape = (20, 20, 16)
        
    """ 
    Lets add the fully connected layer say 120 - we need the shape to be compatible - for that we are adding the
    the dimension of the above layer 
    """
    weight_layer4 =  np.random.rand(output_layer3.shape[0],120,output_layer3.shape[2]) 
    print("Fully Connected Weight4 shape =", weight_layer4.shape) # (20, 120, 16)
    # this time there is no convolution - rather we need to do a dot
    output_layer4 = np.einsum('ijp,jkp->ik', output_layer3, weight_layer4) # (20, 120)
    #output_layer4 = np.tensordot(output_layer3,weight_layer4,axes=2)
    output_layer4 = util.sigmoid(output_layer4)
    print("Fully Connected Layer 1 Ouput shape =", output_layer4.shape)

    weight_layer5 =  np.random.rand(output_layer4.shape[1],1) 
    print("Fully Connected Weight5 shape =", weight_layer5.shape) # (20, 120, 16)
    # this time there is no convolution - rather we need to do a dot
    output_layer5 = np.einsum('ij,jk->ik', output_layer4, weight_layer5) # (20, 120)
    output_layer5 = util.sigmoid(output_layer5)
    print("Fully Connected Layer 2 Ouput shape =", output_layer5.shape)

    
    # final layer lets make it 10 classes
    weight_layer6 =  np.random.rand(output_layer5.shape[0],10) 
    print("Fully Connected Weight6 shape =", weight_layer6.shape) # (20, 120, 16)
    # this time there is no convolution - rather we need to do a dot
    output_layer6 = np.einsum('ij,ik->jk', output_layer5, weight_layer6).flatten() #  20,1*20,10 (1, 10)
    print("Final  Ouput shape =", output_layer6.shape)
    #print("Final  Ouput  =", output_layer6)
    
    """
    Run Softmax
    """
    softmax_ouput =util.softmax(output_layer6)
    print("Softmax  Ouput  =", softmax_ouput)

     # Assume that the truth was class 1 , for this particular "image"
    target = np.array([1., 0., 0., 0., 0. ,0., 0. ,0 ,0., 0.])
     
    #Plug this into the cost function lets take the CrossEntropy Loss as this a classificaiton
    # See this https://www.youtube.com/watch?v=dEXPMQXoiLc  
    # Get index of the true calss
    E_crossEntropyLoss =-np.log(softmax_ouput[np.argmax(target)])
    print("crossEntropyLoss  = ",E_crossEntropyLoss)

    """
    BackPropogate the Loss
    """


    lr = 1 # learning rate
    # https://www.ics.uci.edu/~pjsadows/notes.pdf
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    Sj = softmax_ouput
    DjSi = Si(dij-Sj) #dij = 1 if i == j; else 0

    dE_by_dW6 = (softmax_ouput-target)*output_layer6
    weight_layer6 = weight_layer6 - lr*dE_by_dW6
    t =(output_layer4*(1-output_layer4))
    print(t.shape)
    dE_by_dW5 = (softmax_ouput-target)*weight_layer5