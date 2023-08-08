#include "NNet/NNet.h"
#include "NNet/NNet_samples.h"
#include "Image/Image.h"
using namespace BNN;

int main() {
	//std::string in_folder = "In2/";
	std::string out_folder = "Out2/";
	vector<Image> out; out.reserve(8);
	out.push_back(Image(out_folder + "1.png", 3));
	out.push_back(Image(out_folder + "2.png", 3));
	out.push_back(Image(out_folder + "3.png", 3));
	out.push_back(Image(out_folder + "4.png", 3));
	out.push_back(Image(out_folder + "5.png", 3));
	out.push_back(Image(out_folder + "6.png", 3));
	out.push_back(Image(out_folder + "7.png", 3));
	out.push_back(Image(out_folder + "8.png", 3));
	//Downasmple images
	NNet scl(Downscaler(out[0].pdim(), 2));
	Tenarr x(out.size(), scl.Out_dim(0), scl.Out_dim(1), scl.Out_dim(2));
	for(int i = 0; i < out.size(); i++) x.chip(i, 0) = scl.Compute(out[i].tensor());
#if 1
	vector<Layer*> top;
	//hidden layers
	top.push_back(new Input(scl.Out_dims()));
	top.push_back(new TConv(9, 2, 2, 0, top.back(), false, Afun::t_relu));
	//top.push_back(new Dropout(0.2f, top.back()));
	top.push_back(new Conv(3, 3, 1, 1, top.back(), false, Afun::t_relu));
	top.push_back(new Output(top.back(), Afun::t_sat, Efun::t_mse));;
	auto opt = new Adam(0.003f);
	NNet net(top, opt, "tup9x2_3");
#else
	NNet net("tup9x2_3");
	net.Add_hidden(new Dropout(0.1f, net.Dim_of(0)), 1);
	net.Compile(true);
#endif
	Tenarr y(out.size(), net.Out_dim(0), net.Out_dim(1), net.Out_dim(2));
	for(int i = 0; i < out.size(); i++) y.chip(i, 0) = out[i].tensor();
	for(int i = 0; i < 100; i++) {
		if(!net.Train_parallel(x, y, 8, 100, 100)) break;
		net.Save();
		net.Save_images(x);
	}
}

