#include "pch.h"
using namespace BNN;

//rgb upscaling
int main() {
	constexpr idx train_set = 500;
	constexpr idx test_set = 20;
	//Input data
	std::string in_folder = "Downscaler/";
	Tenarr x(3, 240, 160, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		x.chip(i, 3) = Image(in_folder + std::to_string(i) + ".png", 3).tensor_rgb();
	//Output data
	std::string out_folder = "Reference/";
	Tenarr y(3, 480, 320, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		y.chip(i, 3) = Image(out_folder + std::to_string(i) + ".png", 3).tensor_rgb();
	//Test data
	std::string test_folder = "Test/";
	Tenarr z(3, 240, 160, test_set);
	for(idx i = 0; i < test_set; i++)
		z.chip(i, 3) = Image(test_folder + std::to_string(i) + ".png", 3).tensor_rgb();
#if 1
	//hidden layers
	vector<Layer*> top;
	top.push_back(new Input(shp3(3, 240, 160)));
	//top.push_back(new Conv(12, 5, 1, 2, top.back(), true, Afun::t_lrelu));
	//top.push_back(new Conv(32, 5, 1, 2, top.back(), true, Afun::t_cubl));
	//top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_cubl));
	top.push_back(new Conv(12, 5, 1, 2, top.back(), true, Afun::t_cubl));
	//top.push_back(new Conv(12, 3, 1, 1, top.back(), true, Afun::t_cubl));
	//top.push_back(new Conv(12, 3, 1, 1, top.back(), true, Afun::t_cubl));
	top.push_back(new OutShuf(top.back(), 2));
	auto opt = new Adam(0.001f);
	NNet net(top, opt, "ups_c5_ps2");
#else
	NNet net("ups_c5x32_c3x32_c3_ps2");
	/*net.Set_optim(new RMSprop(0.002f));
	net.Compile();*/
#endif
	for(idx i = 0; i < 100; i++) {
		if(!net.Train_parallel(x, y, 25, 0, 48, 100, 16, 10)) break;
		net.Save();
		net.Save_images(z);
	}
}
