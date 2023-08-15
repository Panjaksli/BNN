#include "pch.h"
using namespace BNN;

int main() {
	constexpr idx train_set = 64;
	constexpr idx test_set = 16;
	//Input data
	std::string in_folder = "Downscaler/";
	Tenarr x(train_set, 3, 160, 240);
	for(idx i = 0; i < train_set; i++)
		x.chip(i, 0) = Image(in_folder + std::to_string(i) + ".png", 3).tensor();
	//Output data
	std::string out_folder = "Reference/";
	Tenarr y(train_set, 3, 320, 480);
	for(idx i = 0; i < train_set; i++) y.chip(i, 0) = Image(out_folder + std::to_string(i) + ".png", 3).tensor();
	//Test data
	std::string test_folder = "Test/";
	Tenarr z(test_set, 3, 160, 240);
	for(idx i = 0; i < test_set; i++) y.chip(i, 0) = Image(test_folder + std::to_string(i) + ".png", 3).tensor();
#if 1
	//hidden layers
	vector<Layer*> top;
	top.push_back(new Input(dim1<3>{3, 160, 240}));
	top.push_back(new Conv(12, 5, 1, 2, top.back(), true, Afun::t_lrelu));
	top.push_back(new Conv(12, 3, 1, 1, top.back(), true, Afun::t_lrelu));
	top.push_back(new OutShuf(top.back(), 2));
	auto opt = new Adam(0.003f);
	NNet net(top, opt, "ups_c5_c3_ps2");
#else
	NNet net("ups_c3_ps2");
#endif
	for(idx i = 0; i < 100; i++) {
		if(!net.Train_parallel(x, y, 16, 0, 100, 100)) break;
		net.Save();
		net.Save_images(z);
	}
}
