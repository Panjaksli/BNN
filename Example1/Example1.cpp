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
	NNet scl(Downscaler(out[0].dim(), 2));
	Tenarr x(out.size(), scl.Out_dim(0), scl.Out_dim(1), scl.Out_dim(2));
	for(int i = 0; i < out.size(); i++) x.chip(i, 0) = scl.Compute(out[i].tensor());
#if 0
	//hidden layers
	vector<Layer*> top;
	top.push_back(new Input(scl.Out_dims()));
	top.push_back(new Conv(12, 5, 1, 2, top.back(), true, Afun::t_lrelu));
	top.push_back(new OutShuf(top.back(), 2));
	auto opt = new Adam(0.005f);
	NNet net(top, opt, "ups_c5_ps2");
#else
	NNet net("ups_c5_ps2");
#endif
	Tenarr y(out.size(), net.Out_dim(0), net.Out_dim(1), net.Out_dim(2));
	for(int i = 0; i < out.size(); i++) y.chip(i, 0) = out[i].tensor();
	for(int i = 0; i < 100; i++) {
		if(!net.Train_parallel(x, y, 16, 0, 100, 100)) break;
		net.Save();
		net.Save_images(x);
	}
}
