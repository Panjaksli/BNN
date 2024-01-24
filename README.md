# BNN
Basic Neural Net -  Neural network built on top of Eigen/Tensor library, offering support for various layers and parallel CPU learning. The library delivers exceptional performance in spatial convolutions, achieving up to 100x performance of Convolve method from Eigen/Tensor (it is very suboptimal).
## What is this library ?
This is an Eigen based sequential neural network library with goal of implementing various layer types, activation functions and optimizers whilst achieving superior perfomance in CPU training **and I really mean superior**. Almost everything is implemented from scratch !
## Performance optimizations
Computations use the bare minimum of temporaries and leverage column major layout of tensors as much as possible (especially the custom spatial convolution algorithm). The number of virtual calls is kept to the absolute minimum too.The library vectorizes quite well and can even leverage multiple CPU cores for learning and inference.
## Features
The Library provides many non-standard features like the previously mentioned **custom convolution algorithm**, tons of custom activation functions mainly **cubic** and **cubic-linear** that aim to replace swish function with superior performance both speed-wise and training-wise.
## Interface
The interface is as simple as possible - create vector<Layer> and push input, hidden layers and output, create optimizer and then pass both to the network, it manages the given memory itself **(dont delete anything manually!)**.
```cpp
using namespace BNN;
vector<Layer*> top;
top.push_back(new Input(shp3(3, 240, 160)));
top.push_back(new Conv(12, 5, 1, 2, top.back(), true, Afun::t_cubl));
top.push_back(new Conv(12, 3, 1, 1, top.back(), true, Afun::t_cubl));
top.push_back(new Conv(12, 3, 1, 1, top.back(), true, Afun::t_cubl));
top.push_back(new OutShuf(top.back(), 2));
//Optimizer with: learning rate, regularizer 
auto opt = new Adam(0.001f, Regular(RegulTag::L2, 0.1f));
NNet net(top, opt, "Network name");
```
If you do any changes to the architecture after that, you need to compile it before running !\
For training you can then use the Train_single(), Train_parallel() or **Train_Minibatch()** functions, followed by Save() to save the network to a hybrid text/binary file.
Save_images() can be used to save the output tensor as png image(s), if it has 1 or 3 channels.
```cpp
//Channels, Width, Height, Count
Tenarr x(3, 240, 160, train_set);
Tenarr y(3, 240, 160, train_set);
for(int i = 0; i < 100; i++) {
  // input, target, epochs, minibatch size, threads, steps (minibatches per epoch), learning rate
	if(!net.Train_Minibatch(x, y, 20, 16, 16, -1, decay_rate(0.001f, i, 20))) break;
	net.Save();
	net.Save_images(z);
}
```
See Example1 or Example2 for further usage !!!
## Comparison relu vs cubic-linear vs swish
### Training cost in 2 layer upscaling CNN:
![image](https://github.com/Panjaksli/BNN/assets/82727531/41292bdb-1f6f-4afc-a447-e4f843288343)
Loss-wise relu is the best, cubl slightly outperforms swish.
### Image quality:
![image](https://github.com/Panjaksli/BNN/assets/82727531/e066678c-629e-4c8d-99d4-abff40ee6de3)
Quality-wise cubl and swish provide smoother upscaled image than relu.
## Custom pre-trained CNN for image upscaling
This repository comes with custom pre-trained model for high quality image upscaling, that achieves far better results than any simple upscaling algorithm (bicubic, bilinear). And even though the network was trained only on real-life images, it performs on upscaling anime art too !
### Comparison: bicubic vs model vs reference
![image](https://github.com/Panjaksli/BNN/assets/82727531/fb3a9592-5987-4eb9-bde0-dccecb1c459e)
### Upscaling anime art
![image](https://github.com/Panjaksli/BNN/assets/82727531/718568a6-111a-4436-870b-c206874185eb)
### How does it work ?
The model is trained on the error of reference image and low res image upscaled with bicubic interpolation:\
d(x) = f(x) - g(x),\
where: d(x) is error function, f(x) is full resolution image and g(x) is an approximation of f(x).\
This error is then added to the upscaled image during inference:\
f(x) = g(x) + d(x) = g(x) - g(x) + f(x) = f(x).\
This results in reconstruction of original image f(x).

