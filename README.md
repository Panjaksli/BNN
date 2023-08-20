# BNN
Basic Neural Net -  Neural network built on top of Eigen/Tensor library, offering support for various layers and parallel CPU learning. The library delivers exceptional performance in spatial convolutions, achieving up to 100x performance of Convolve method from Eigen/Tensor along with much better CPU clock scaling
## What is this library ?
This is an Eigen based sequential neural network library with goal of implementing various layer types, activation functions and optimizers whilst achieving superior perfomance in CPU training **and I really mean superior**.
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
auto opt = new Adam(0.001f);
NNet net(top, opt, "Network name");
```
If you do any changes to the architecture after that, you need to compile it before running !\
For training you can then use the Train_single() or Train_parallel() functions, followed by Save() to save the network to a hybrid text/binary file.
Save_images() can be used to save the output tensor as png image(s).
```cpp
for(int i = 0; i < 100; i++) {
  // input output, epochs, rate (0 = default), batch size, log count, threads, steps (each step shuffles dataset) 
	if(!net.Train_parallel(x, y, 20, 0, 48, 100, 16, 10)) break;
	net.Save();
	net.Save_images(z);
}
```
## Comparison lrelu vs cubic-linear vs swish
### Training cost in 2 layer upscaling CNN:
![image](https://github.com/Panjaksli/BNN/assets/82727531/da8af6b2-96b3-4212-8a4c-4a1fa411c5cb)
As you can see, the cubl and swish are outperforming lrelu by miles. Cubl has a tiny bit lower training cost than swish.
### Image quality:
![image](https://github.com/Panjaksli/BNN/assets/82727531/1172d914-1f6e-4c2a-bbfa-5f84bfe1f87a)
Swish and cubl dont suffer the same horrible artifacts as lrelu. Cubl achieves a tiny bit better local contrast than swish (eg. the text on the sign)
