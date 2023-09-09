#include "pch.h"
using namespace BNN;
//rgb upscaling
int main() {
	#if 0
		//Inference code
		NNet net("Upscaling_net");
		if(!net.Valid()) return 1;
		std::string inp, outp; int fact = 2;
		while(1) {
			print("Input: "); std::cin >> inp;
			if(inp == "exit") return 0;
			print("Output: "); std::cin >> outp;
			print("Factor (multiple of 2): "); std::cin >> fact;
			Tensor in = Image(inp, 3, 1).tensor_rgb(true);
			if(in.size() == 0) { println("Invalid image!"); continue; }
			if(outp.empty()) { println("Invalid output!"); continue; }
			if(fact <= 0) { fact = 1; }
			Tensor out = in;
			for(int i = 1; i < fact; i *= 2) {
				out = net.Compute_DS(out) + resize(out, 2, 2);
			}
			Image(out).save(outp + ".png");
		}
	#else
		//Training code
		constexpr idx train_set = 500;
		constexpr idx test_set = 20;
		std::string parent = "Upscaler/";
		//Input data
		std::string in_folder = "Downscaler/";
		std::string netname = parent + "c3x32x4";
		Tenarr x(3, 240, 160, train_set);
	#pragma omp parallel for
		for(idx i = 0; i < train_set; i++)
			x.chip(i, 3) = Image(parent + in_folder + std::to_string(i), 3).tensor_rgb();
		//Output data
		std::string out_folder = "Reference/";
		Tenarr y(3, 480, 320, train_set);
	#pragma omp parallel for
		for(idx i = 0; i < train_set; i++)
			y.chip(i, 3) = Image(parent + out_folder + std::to_string(i), 3).tensor_rgb() - resize(x.chip(i, 3), 2, 2);
		//Test data
		std::string test_folder = "Test/";
		Tenarr z(3, 240, 160, test_set);
	#pragma omp parallel for
		for(idx i = 0; i < test_set; i++)
			z.chip(i, 3) = Image(parent + test_folder + std::to_string(i), 3).tensor_rgb();
		/*for(idx i = 0; i < test_set; i++)
		Image(10.f * (Image(parent + "Test_ref/" + std::to_string(i), 3).tensor_rgb() - resize(z.chip(i, 3), 2, 2) ).abs()).save(parent + "Test_dy/" + std::to_string(i) + ".png");*/
	#if 1
		//hidden layers
		vector<Layer*> top;
		top.push_back(new Input(shp3(3, 240, 160)));
		//top.push_back(new Conv(64, 5, 1, 2, top.back(), true, Afun::t_cubl));
		top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_cubl));
		top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_cubl));
		top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_cubl));
		top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_cubl));
		top.push_back(new Conv(12, 3, 1, 1, top.back(), true, Afun::t_clu));
		/*top.push_back(new SConv(3, 1, 1, top.back()));
		top.push_back(new Conv(32, 1, 1, 0, top.back(), true, Afun::t_cubl));*/
		top.push_back(new OutShuf(top.back(), 2, Efun::t_mae));
		auto opt = new Adam(0.002f, 0.2f, L2);
	
		NNet net(top, opt, netname);
	#else
		NNet net(netname, true);
	#endif
		for(idx i = 0; i < 100; i++) {
	
			if(!net.Train_parallel(x, y, 30, 0.001, 16, 100, 16, 5)) break;
			net.Save();
			for(idx j = 0; j < 10; j++)
				Image(net.Compute(z.chip(j, 3)).abs() * 10.f).save(netname + "/" + std::to_string(j) + ".png");
			for(idx j = 10; j < test_set; j++)
				Image(net.Compute(z.chip(j, 3)) + resize(z.chip(j, 3), 2, 2)).save(netname + "/" + std::to_string(j) + ".png");
			//for(idx j = 0; j < test_set; j++)
			//	Image(0.5f * (net.Compute(z.chip(j, 3)) + 1.f) /* + resize(z.chip(j, 3), 2, 2)*/).save(netname + "/" + std::to_string(j) + ".png");
			// + bil_ups(z.chip(j, 3), 2, 2)
	
			//net.Save_images(z);
		}
	#endif
}
