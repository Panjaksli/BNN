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
		y.chip(i, 3) = Image(parent + out_folder + std::to_string(i), 3).tensor_rgb() - x.chip(i, 3);
	std::string netname = parent + "c5x32_c3x32_c3";
#if 0
	//hidden layers
	vector<Layer*> top;
	top.push_back(new Input(shp3(3, 480, 360)));
	top.push_back(new Conv(32, 5, 1, 2, top.back(), true, Afun::t_cubl));
	top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_cubl));
	top.push_back(new Conv(3, 3, 1, 1, top.back(), true, Afun::t_clu));
	top.push_back(new Output(top.back(), Efun::t_mae));
	auto opt = new Adam(0.002f, Regular(RegulTag::L2, 0.01f));
	NNet net(top, opt, netname);
#else
	NNet net(netname);
#endif
	for(idx i = 0; i < 100; i++) {
		if(!net.Train_parallel(x, y, 20, 0.001, -1, 100, train_set, 1)) break;
		net.Save();
#pragma omp parallel for
		for(idx j = 0; j < train_set; j++)
			Image(net.Compute(x.chip(j, 3)) + x.chip(j, 3)).save(netname + "/" + std::to_string(j) + ".png");
	}
}
