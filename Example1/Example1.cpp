#include "pch.h"
using namespace BNN;

//rgb upscaling
int main() {
	Tensor pp(5, 10, 8);
	pp.setRandom();
	dim1<3> maxidx;
	pp.maximum(maxidx);
	constexpr idx train_set = 500;
	constexpr idx test_set = 20;
	//Test data
	std::string test_folder = "Test/";
	Tenarr z(3, 240, 160, test_set);
	for(idx i = 0; i < test_set; i++)
		z.chip(i, 3) = Image(test_folder + std::to_string(i), 3).tensor_rgb();
	//Input data
	std::string in_folder = "Downscaler/";
	Tenarr x(3, 240, 160, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		x.chip(i, 3) = Image(in_folder + std::to_string(i), 3).tensor_rgb();
	//Output data
	std::string out_folder = "Reference/";
	Tenarr y(3, 480, 320, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		y.chip(i, 3) = Image(out_folder + std::to_string(i), 3).tensor_rgb();

#if 0
	//hidden layers
	vector<Layer*> top;
	top.push_back(new Input(shp3(3, 240, 160)));
	top.push_back(new Resize(top.back(),2));
	//top.push_back(new Conv(32, 5, 1, 2, top.back(), true, Afun::t_cubl));
	//top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_cubl));
	//top.push_back(new Conv(12, 3, 1, 1, top.back(), true, Afun::t_cubl));
	//top.push_back(new Conv(16, 5, 1, 2, top.back(), true, Afun::t_cubl));
	//top.push_back(new Conv(16, 3, 1, 1, top.back(), true, Afun::t_cubl));
	top.push_back(new Conv(3, 5, 1, 2, top.back(), true, Afun::t_cubl));
	top.push_back(new Conv(3, 3, 1, 1, top.back(), true, Afun::t_cubl));
	top.push_back(new Output(top.back(), Efun::t_mae));
	//top.push_back(new OutShuf(top.back(), 2, Efun::t_mae));
	auto opt = new Adam(0.001f, 0.01f, L1);
	NNet net(top, opt, "ups_bil_c5_c3_L1");
#else
	NNet net("ups_c5_c3_ps2_L1");
#endif
	for(idx i = 0; i < 100; i++) {
		if(!net.Train_parallel(x, y, 50, 0.001, 80, 100, 16, 5)) break;
		net.Save();
		net.Save_images(z);
	}
}
