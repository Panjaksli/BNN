#include "pch.h"
using namespace BNN;

int main() {
	constexpr idx train_set = 500;
	constexpr idx test_set = 16;
	//Input data
	std::string in_folder = "Downscaler/";
	Tenarr x(train_set, 3, 160, 240);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		x.chip(i, 0) = Image(in_folder + std::to_string(i) + ".png", 3).tensor();
	//Output data
	std::string out_folder = "Reference/";
	Tenarr y(train_set, 3, 320, 480);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		y.chip(i, 0) = Image(out_folder + std::to_string(i) + ".png", 3).tensor();
	//Test data
	std::string test_folder = "Test/";
	Tenarr z(test_set, 3, 160, 240);
	for(idx i = 0; i < test_set; i++)
		z.chip(i, 0) = Image(test_folder + std::to_string(i) + ".png", 3).tensor();
#if 1
	//hidden layers
	vector<Layer*> top;
	top.push_back(new Input(dim1<3>{3, 160, 240}));
	//top.push_back(new Conv(32, 5, 1, 2, top.back(), true, Afun::t_lrelu));
	//top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_lrelu));
	top.push_back(new Conv(12, 3, 1, 1, top.back(), true, Afun::t_lrelu));
	top.push_back(new OutShuf(top.back(), 2));
	auto opt = new Adam(0.001f);
	NNet net(top, opt, "ups_c3_ps2");
#else
	NNet net("ups_c5x32_c3x32_c3_ps2");
	//net.Set_optim(new Adam(0.002f));
	//net.Compile();
#endif
	for(idx i = 0; i < 100; i++) {
		if(!net.Train_parallel(x, y, 20, 0.0005f, 80, 100, 16,10)) break;
		net.Save();
		net.Save_images(z);
	}
}
