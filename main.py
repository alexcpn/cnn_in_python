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

def initializeWeights(number_of_filters,filter_size,depth):
    """Initialize the weight's/filters of Layer
    :return: list of weights
    """
    weight_layer = []
    for i in range(number_of_filters):
        weight =  np.random.rand(filter_size,filter_size,depth) 
        weight_layer.append(weight)
    return weight_layer

def layerConvolutionActivation(image, weight_layer1):
    """This function  does the Convolution with the filter and then applies the
    activation function to the convolution 
    :return: the output layer after applying the Activation
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
    #print("Activation Shape after layer 1=",output_layer2.shape)
    return output_layer2

if __name__ == '__main__':
    
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    # Generate a random image
    image_size = 32 
    image_depth = 3
    image = np.random.rand(image_size, image_size)
    # to mimic RGB channel
    image = np.stack([image,image,image], axis=image_depth-1) # 0 to 2
    print("Image Shape=",image.shape)
    print("Image [0,0,:]=",image[0,1,2])

    # The class containing the convolution Logic
    testConv2D = cnn.Conv2D()

    # we will create leNet without the Pooling parts
    # (stride is always 1 for now)
    #  [32.32.3] *(5.5.3)*6  == [28.28.6] * (5.5.6)*1 = [24.24.1] *  (5.5.3)*16 = [20.20.16] * FC layer 
    
    print("-----------Forward Propogation------------------------")

    # For layer 1
    filter_size = 5  
    number_of_filters = 6
    # Initialize the weight's/filters of Layer1
    w1 = initializeWeights(number_of_filters,filter_size,image.shape[2])
    a1 = layerConvolutionActivation(image,w1)
    print("w1.len =", len(w1))
    print("w1[0].shape =", w1[0].shape)
     # For layer 2
    filter_size = 5  
    number_of_filters = 1
    # Initialize the weight's/filters of Layer1
    w2 =  initializeWeights(number_of_filters,filter_size,a1.shape[2])
    # Do convolution and activation
    a2 = layerConvolutionActivation(a1,w2)
    print("w2.len =", len(w2))
    print("w2[0].shape =", w2[0].shape)
    # For layer 3
    # Out=W???F+1 imagesize - filtersize + 1
    filter_size = 5  
    number_of_filters = 16
    # Initialize the weight's/filters of Layer1
    w3 =  initializeWeights(number_of_filters,filter_size,a2.shape[2])
    print("w3.len =", len(w3))
    print("w3[0].shape =", w3[0].shape)
    # Do convolution and activation
    a3 = layerConvolutionActivation(a2,w3)
    print("a3.shape=", a3.shape) # output_layer3 shape = (20, 20, 16)
        
    """ 
    Lets add the fully connected layer say 120 - we need the shape to be compatible - for that we are adding the
    the dimension of the above layer 
    """
    w4 =  np.random.rand(a3.shape[0],120,a3.shape[2]) 
    print("w4.shape =", w4.shape) # (20, 120, 16)
    # this time there is no convolution - rather we need to do a dot
    z4 = np.einsum('ijp,jkp->ik', a3, w4) # (20, 120)
    #output_layer4 = np.tensordot(output_layer3,weight_layer4,axes=2)
    print("z4.shape =", z4.shape)
    a4 = util.sigmoid(z4)
    print("a4.shape =", a4.shape)

    w5 =  np.random.rand(a4.shape[1],1) 
    print("w5.shape =", w5.shape) # (20, 120, 16)
    # this time there is no convolution - rather we need to do a dot
    z5 = np.einsum('ij,jk->ik', a4, w5) # (20, 120)
    print("z5.shape =", z5.shape)
    a5 = util.sigmoid(z5)
    print("a5.shape =", a5.shape)
    
    # final layer lets make it 10 classes
    w6 =  np.random.rand(a5.shape[0],10) 
    print("w6.shape =", w6.shape) # (20, 120, 16)
    # this time there is no convolution - rather we need to do a dot
    logits = np.einsum('ij,ik->jk', a5, w6).flatten() #  20,1*20,10 (1, 10) == Z_l
    z6 = logits
    print("z6.shape=", z6.shape) #(10,)
    #print("Final  Output  =", output_layer6)
    
    """
    Run Softmax
    """
    softmax_ouput =util.softmax(logits)
    a6 = softmax_ouput
    print("a6.shape =", a6.shape)
    print("Softmax Output  =", softmax_ouput)

     # Assume that the truth was class 1 , for this particular "image"
    target = np.array([1., 0., 0., 0., 0. ,0., 0. ,0 ,0., 0.])
     
    #Plug this into the cost function lets take the CrossEntropy Loss as this a classification
    # See this https://www.youtube.com/watch?v=dEXPMQXoiLc  
    # Get index of the true class
    LcrossEntropyLoss = util.crossentropyloss(softmax_ouput,target)
    print("crossEntropyLoss  = ",LcrossEntropyLoss) # ex 3.75671634607845
    
    """
    Back-Propagate the Loss
    """
    print("-----------Back Propogation------------------------")

    lr = 1 # learning rate
    # https://alexcpn.github.io/html/NN/ml/7_backpropogation_full/
    D_S_by_z = util.derv_softmax_wrto_logits(softmax_ouput)
    D_L_by_z = util.derv_crossentropyloss_wrto_logits(softmax_ouput,target)
    # For the last layers  W= 6
    # EqA1 in https://alexcpn.github.io/html/NN/ml/7_backpropogation_full/
    activation_L =  softmax_ouput
    DL_by_z6 = (activation_L -target)
    print("DL_by_z6.shape =", DL_by_z6.shape)
    D_L_by_w6 = DL_by_z6 *a5
    print("BP 6: Last weight update - D_L_by_w6 shape == w6 shape",D_L_by_w6.shape,w6.shape)
    w6 = w6 - lr*D_L_by_w6
    

    print("-----------------------------------")
    # For the inner layers  W= 5
    # EqA2 in https://alexcpn.github.io/html/NN/ml/7_backpropogation_full/
    a5_derv = util.derv_sigmoid(z5)
    #DL_by_z6 = np.expand_dims(DL_by_z6, axis=1)
    DL_by_z5 = DL_by_z6 @ w6.T  # @ is np.matmul (almost similar to np.dot -https://stackoverflow.com/a/34142617/429476 )
    DL_by_z5 = DL_by_z5 * a5_derv #  Hadamard product/element wise multiplication here (20, 1) * (20, 1)
    print("DL_by_z5 shape",DL_by_z5.shape) # (20, 1)
    D_L_by_w5 =   a4.T @ DL_by_z5
    print("D_L_by_w5 shape == w5 Shape",D_L_by_w5.shape,w5.shape)
   
    # Using gradient descent to update the weights
    print("BP 5: Last weight update")
    w5 = w5 - lr*D_L_by_w5
    print("w5 shape after update", w5.shape)
   
    print("-----------------------------------")
    # For the inner layers  W= 4
    a4_derv = util.derv_sigmoid(z4)
    DL_by_z4 = DL_by_z5 @ w5.T
    DL_by_z4 = DL_by_z4 * a4_derv #  Hadamard product/element wise multiplication here (20, 120) * (20, 120)
    print("DL_by_z4 shape",DL_by_z4.shape) # (20, 120)
    print("a3 shape",a3.shape)
    #D_L_by_w4 = a3.T @ DL_by_z4 #(20,20,16).T @ (20,120) ==> (16,20,120); we need 20,120,16; need np.Transpose or below
    D_L_by_w4=np.einsum('ijp,ik->ikp', a3, DL_by_z4) 
    print("D_L_by_w4 shape == w4 Shape",D_L_by_w4.shape,w4.shape)
    # Using gradient descent to update the weights
    print("BP 4: Last weight update")
    w4 = w4 - lr*D_L_by_w4
    


    # Other References

    # https://e2eml.school/softmax.html
    # https://stats.stackexchange.com/a/564725/191675
    # https://bfeba431-a-62cb3a1a-s-sites.googlegroups.com/site/deeplearningcvpr2014/ranzato_cvpr2014_DLtutorial.pdf?attachauth=ANoY7cqPhkgQyNhJ9E7rmSk-RTdMYSYqpfJU2gPlb9cWH_4a1MbiYPq_0ihyuolPiYDkImyr9PmA-QwSuS8F3OMChiF97XTDD_luJqam70GvAC4X6G6KlU2r7Pv1rqkHaMbmXpdtXJHAveR_jWf1my_IojxFact87u2-1YXtfJIwYkhBwhMsYagICk-P6X9ktA0Pyjd601tboSlX_UGftX1vB57-tS6bdAkukhmSRLU-ZiF4RdJ_sI3YAGaaPYj1KLWFpkFa_-XG&attredirects=1
    # https://cs.nyu.edu/~yann/talks/lecun-ranzato-icml2013.pdf
