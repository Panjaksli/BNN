#include "pch.h"
using namespace BNN;

//rgb upscaling
int main() {
#if 0
	//Inference code
	NNet net("Upscaling_net");
	std::string inp, outp;
	while(1) {

		print("Input: "); std::cin >> inp;
		if(inp == "exit") return 0;
		print("Output: "); std::cin >> outp;
		//RGB MODEL
		//Image(net.Compute_DS(Image(inp, 3, 1))).save(outp + ".png");
		//YUV MODEL
		/*Tensor x = Image(inp, 3, 1);
		Tensor y(3, x.dimension(1) * 2, x.dimension(2) * 2);
		for(int i = 0; i < 3; i++)
			y.chip(i, 0) = net.Compute_DS(x.chip(i, 0).reshape(dim1<3>{1, x.dimension(1), x.dimension(2)})).chip(i, 0);
		Image(y).save(outp + ".png");*/
		//DIFF MODEL
		Tensor x = Image(inp, 3, 1).tensor_rgb();
		Image(net.Compute_DS(x) + resize(x, 2, 2)).save(outp + ".png");
		//Cubic
		//Image(inp, 3, 1).resize_cubic(960,720).save(outp + ".png");
	}
#else
	//Training code
	constexpr idx train_set = 500;
	constexpr idx test_set = 20;
	std::string parent = "Upscaler/";
	//Input data
	std::string in_folder = "Downscaler/";
	std::string netname = parent + "ups_c5x32_c3x32x2_c3_ps2";
	Tenarr x(3, 240, 160, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		x.chip(i, 3) = Image(parent + in_folder + std::to_string(i), 3).tensor_rgb();
	//Output data
	std::string out_folder = "Reference/";
	Tenarr y(3, 480, 320, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		y.chip(i, 3) = Image(parent + out_folder + std::to_string(i), 3).tensor_rgb();// -resize(x.chip(i, 3), 2, 2);
	//Test data
	std::string test_folder = "Test/";
	Tenarr z(3, 240, 160, test_set);
#pragma omp parallel for
	for(idx i = 0; i < test_set; i++)
		z.chip(i, 3) = Image(parent + test_folder + std::to_string(i), 3).tensor_rgb();
	//for(idx i = 0; i < 4; i++)
	//	Image(parent + test_folder + std::to_string(i), 3).resize(480,320,Linear).save(parent + "Test_scl/" + std::to_string(i) + ".png");
	//	Image((resize(z.chip(i, 3),2,2,Linear) - Image(parent + "Test_ref/" + std::to_string(i), 3).tensor_rgb()).abs()).save(parent + "Test_dy_li/" + std::to_string(i) + ".png");
#if 1
	//hidden layers
	vector<Layer*> top;
	top.push_back(new Input(shp3(3, 240, 160)));
	top.push_back(new Conv(32, 5, 1, 2, top.back(), true, Afun::t_cubl));
	top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_cubl));
	top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_cubl));
	//top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_cubl));
	top.push_back(new Conv(12, 3, 1, 1, top.back(), true, Afun::t_cubl));
	top.push_back(new OutShuf(top.back(), 2, Efun::t_mae));
	//top.push_back(new Output(top.back(), Efun::t_mae));
	auto opt = new Adam(0.001f, 0.2f, L2);

	NNet net(top, opt, netname);
#else
	NNet net(netname);
#endif
	for(idx i = 0; i < 100; i++) {
		if(!net.Train_parallel(x, y, 50, 0.003, 80, 100, 16, 5)) break;
		net.Save();
		for(idx j = 0; j < 20; j++)
			Image(net.Compute(z.chip(j, 3))/* + resize(z.chip(j, 3), 2, 2)*/).save(netname + "/" + std::to_string(j) + ".png");
		// + bil_ups(z.chip(j, 3), 2, 2)

		//net.Save_images(z);
	}
#endif
}
