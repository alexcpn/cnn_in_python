Aim: Create a CNN NN from Python to learn
Use no libraries other than for data loading and image manipulation 

DataSet - https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

Terminal Ouput

```
python3 main.py
Image Shape= (32, 32, 3)
Image [0,0,:]= 0.828110384728807
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
a5_derv.shape = (20, 1)
w6.shape = (20, 10)
z6.shape= (10,)
a6.shape = (10,)
Softmax  Ouput  = [0.03 0.10 0.03 0.04 0.27 0.11 0.12 0.06 0.02 0.22]
crossEntropyLoss  =  3.686443962296525
D_L_by_w6 shape (20, 10)
-----------------------------------
D_I_by_wI_1 shape (20,)
D_I_by_w5 shape (120, 1)
w5 shape (120, 1)
w5 shape (120, 1)
-----------------------------------
y shape (20, 1) (1, 120)
D_I_by_wI_2 shape (20, 120)
D_I_by_wI_2 shape (20, 120, 16)
w4 shape (20, 120, 16)
```