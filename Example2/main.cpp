#include "pch.h"
using namespace BNN;

//rgb upscaling
int main() {
	constexpr idx train_set = 12;
	std::string parent = "Denoiser/";

	//Input data
	std::string in_folder = "Input/";
	Tenarr x(3, 480, 360, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		x.chip(i, 3) = Image(parent + in_folder + std::to_string(i), 3).tensor_rgb();
	//Output data
	std::string out_folder = "Reference/";
	Tenarr y(3, 480, 360, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		y.chip(i, 3) = Image(parent + out_folder + std::to_string(i), 3).tensor_rgb();
#if 0
	//hidden layers
	vector<Layer*> top;
	top.push_back(new Input(shp3(3, 480, 360)));
	top.push_back(new Conv(32, 5, 1, 2, top.back(), true, Afun::t_cubl));
	top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_cubl));
	top.push_back(new Conv(3, 3, 1, 1, top.back(), true, Afun::t_cubl));
	top.push_back(new Output(top.back(), Efun::t_mse));
	auto opt = new Adam(0.001f, 0.01f, L2);
	NNet net(top, opt, parent + "den_c5x32_c3x32_c3_L2");
#else
	NNet net(parent + "den_c5x32_c3x32_c3_L2");
#endif
	for(idx i = 0; i < 100; i++) {
		if(!net.Train_parallel(x, y, 50, 0.001, -1, 100, 12, 1)) break;
		net.Save();
		net.Save_images(x);
	}
}
