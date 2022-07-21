
# Aim: Create a CNN NN from Python to learn Back-propagation

Use no libraries other than for data loading and image manipulation 

DataSet - https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
Explanation https://alexcpn.github.io/html/NN/ml/7_backpropogation_full/
(Permanent) https://github.com/alexcpn/alexcpn.github.io/blob/master/html/NN/ml/7_backpropogation_full.md

Terminal Output

```
python3 main.py
Image Shape= (32, 32, 3)
Image [0,0,:]= 0.33449773440189534
-----------Forward Propogation------------------------
Convolution Shape = (28, 28, 6)
Convolution Shape = (24, 24, 1)
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
Softmax Output  = [0.10 0.08 0.16 0.05 0.18 0.21 0.03 0.02 0.11 0.06]
crossEntropyLoss  =  2.3135097712620363
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

