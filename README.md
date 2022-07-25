
# Aim: Create a CNN NN from Python to learn Back-propagation

Use no libraries other than for data loading and image manipulation 

DataSet - https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
Explanation https://alexcpn.github.io/html/NN/ml/7_backpropogation_full/
(Permanent) https://github.com/alexcpn/alexcpn.github.io/blob/master/html/NN/ml/7_backpropogation_full.md

We will create leNet without the Pooling parts

Stride is always 1 for now)

```
Input (R,G,B)= [32.32.3] *(5.5.3)*6  == [28.28.6] * (5.5.6)*1 = [24.24.1] *  (5.5.1)*16 = [20.20.16] *
FC layer 1 (20, 120, 16) * FC layer 2 (120, 1) * FC layer 3 (20, 10) * Softmax  (10,) =(10,1) = Output
```

Terminal Output

```
python3 main.py
Image Shape= (32, 32, 3)
Image [0,0,:]= 0.17265938674851355
-----------Forward Propogation------------------------
Convolution Shape = (28, 28, 6)
w1.len = 6
w1[0].shape = (5, 5, 3)
Convolution Shape = (24, 24, 1)
w2.len = 1
w2[0].shape = (5, 5, 6)
w3.len = 16
w3[0].shape = (5, 5, 1)
Convolution Shape = (20, 20, 16)
a3.shape= (20, 20, 16)
w4.shape = (20, 120, 16)
z4.shape = (20, 120)
a4.shape = (20, 120)
w5.shape = (120, 1)
z5.shape = (20, 1)
a5.shape = (20, 1)
w6.shape = (20, 10)
z6.shape= (10,)
a6.shape = (10,)
Softmax Output  = [0.10 0.02 0.03 0.12 0.02 0.04 0.02 0.55 0.00 0.10]
crossEntropyLoss  =  2.2953522438843943
-----------Back Propogation------------------------
1
DL_by_z6.shape = (10,)
BP 6: Last weight update - D_L_by_w6 shape == w6 shape (20, 10) (20, 10)
-----------------------------------
DL_by_z5 shape (20, 20)
D_L_by_w5 shape == w5 Shape (120, 20) (120, 1)
BP 5: Last weight update
w5 shape after update (120, 20)
-----------------------------------
DL_by_z4 shape (20, 120)
a3 shape (20, 20, 16)
D_L_by_w4 shape == w4 Shape (20, 120, 16) (20, 120, 16)
BP 4: Last weight update
```

