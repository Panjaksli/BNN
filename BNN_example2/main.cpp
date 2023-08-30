#include "pch.h"
using namespace BNN;

//rgb upscaling
int main() {
	Tensor pp(5, 10, 8);
	pp.setRandom();
	dim1<3> maxidx;
	pp.maximum(maxidx);
	constexpr idx train_set = 1;
	//Test data
	std::string in_folder = "den/";
	Tenarr x(3, 960, 720, train_set);
	x.chip(0, 3) = Image(in_folder + "in.png", 3).tensor_rgb();
	//Output data
	std::string out_folder = "den/";
	Tenarr y(3, 960, 720, train_set);
	y.chip(0, 3) = Image(in_folder + "out.png", 3).tensor_rgb();

#if 1
	//hidden layers
	vector<Layer*> top;
	top.push_back(new Input(shp3(3, 960, 720)));
	top.push_back(new Conv(8, 5, 1, 2, top.back(), true, Afun::t_cubl));
	top.push_back(new Conv(8, 3, 1, 1, top.back(), true, Afun::t_cubl));
	top.push_back(new Conv(3, 3, 1, 1, top.back(), true, Afun::t_cubl));
	top.push_back(new Output(top.back(), Efun::t_mae));
	auto opt = new Adam(0.003f, 0.0, L0);
	NNet net(top, opt, "den_c5x8_c3x8_c3_L1");
#else
	NNet net("ups_c5_c3_ps2_L1");
#endif
	for(idx i = 0; i < 100; i++) {
		if(!net.Train_single(x, y, 100)) break;
		net.Save();
		net.Save_images(x);
	}
}
