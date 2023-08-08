#include "pch.h"
using namespace BNN;
int main() {
	std::string folder = "Data/";
	vector<Image> out; out.reserve(16);
	out.push_back(Image(folder + "1.png", 3));
	out.push_back(Image(folder + "2.png", 3));
	out.push_back(Image(folder + "3.png", 3));
	out.push_back(Image(folder + "4.png", 3));
	out.push_back(Image(folder + "5.png", 3));
	out.push_back(Image(folder + "6.png", 3));
	out.push_back(Image(folder + "7.png", 3));
	out.push_back(Image(folder + "8.png", 3));
	out.push_back(Image(folder + "9.png", 3));
	out.push_back(Image(folder + "10.png", 3));
	//Downasmple images
	NNet scl(Downscaler(out[0].pdim(), 2));
	Tenarr x(out.size(), scl.Out_dim(0), scl.Out_dim(1), scl.Out_dim(2));
	for(int i = 0; i < out.size(); i++) x.chip(i, 0) = scl.Compute(out[i].tensor());
	for(int i = 0; i < out.size(); i++) scl.Save_images(x);
#if 1
	//hidden layers
	vector<Layer*> top;
	top.push_back(new Input(scl.Out_dims()));
	top.push_back(new TConv(6, 2, 2, 0, top.back(), false));
	top.push_back(new Dropout(0.1f,top.back()));
	top.push_back(new Conv(3, 3, 1, 1, top.back(), false));
	top.push_back(new Output(top.back(), Afun::t_lin, Efun::t_mse));;
	auto opt = new Adam(0.01f);
	NNet net(top, opt, "ups2x6_d_3");
#else
	NNet net("ups2_3");
	net.Compile(true);
#endif
	Tenarr y(out.size(), net.Out_dim(0), net.Out_dim(1), net.Out_dim(2));
	for(int i = 0; i < out.size(); i++) y.chip(i, 0) = out[i].tensor();
	for(int i = 0; i < 100; i++) {
		if(!net.Train_parallel(x, y, 16, 100, 100)) break;
		net.Save();
		net.Save_images(x);
	}
}