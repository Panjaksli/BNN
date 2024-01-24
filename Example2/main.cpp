#include "pch.h"
using namespace BNN;

int main() {
#if 0
	
	std::string inp, outp;
	while(1) {
		NNet net("DenNet");
		if(!net.Valid()) return 1;
		print("Input: "); std::cin >> inp;
		if(inp == "exit") return 0;
		print("Output: "); std::cin >> outp;
		Tensor in = Image(inp, 3, 1).tensor_rgb(true);
		if(in.size() == 0) { println("Invalid image!"); continue; }
		if(outp.empty()) { println("Invalid output!"); continue; }
		Tensor out = net.Compute_DS(in);
		Image(out).save(outp + ".png");
	}
#else
	constexpr idx train_set = 32;
	std::string parent = "Denoiser/";

	//Input data
	std::string in_folder = "in/";
	Tenarr x(3, 480, 360, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		x.chip(i, 3) = Image(parent + in_folder + std::to_string(i), 3).tensor_rgb();
	//Output data
	std::string out_folder = "out/";
	std::string delta = "Delta/";
	Tenarr y(3, 480, 360, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++) {
		y.chip(i, 3) = Image(parent + out_folder + std::to_string(i), 3).tensor_rgb();// -x.chip(i, 3);
		//Image(y.chip(i, 3).abs()).save(parent + delta + "/" + std::to_string(i) + ".png");
	}
	std::string netname = parent + "c5x32_c3x32x2";
#if 0
	//hidden layers
	vector<Layer*> top;
	top.push_back(new Input(shp3(3, 480, 360)));
	top.push_back(new Conv(32, 5, 1, 2, top.back(), true, Afun::t_lrelu));
	//top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_lrelu));
	top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_lrelu));
	top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_lrelu));
	top.push_back(new Conv(3, 3, 1, 1, top.back(), true, Afun::t_lrelu));
	top.push_back(new Output(top.back(), Efun::t_mae));
	auto opt = new Adam(0.002f, Regular(RegulTag::L2, 0.1f));
	NNet net(top, opt, netname);
#else
	NNet net(netname);
#endif
	constexpr idx rep = 100;
	constexpr idx epoch = 100;
	for(idx i = 0; i < rep; i++) {
		if(!net.Train_Minibatch(x, y, 20, 16, 16, -1, decay_rate(0.002f, i, 20))) break;
		//if(!net.Train_parallel(x, y, 10, 0.002, -1, 100, train_set, 10, true)) break;
		net.Save();
#pragma omp parallel for
		for(idx j = 10; j < train_set; j++) {
			//if(j < train_set / 2)
			Image(net.Compute(x.chip(j, 3)).abs()).save(netname + "/" + std::to_string(j) + ".png");
			//else
			//	Image(net.Compute(x.chip(j, 3)) + x.chip(j, 3)).save(netname + "/" + std::to_string(j) + ".png");
		}
	}
//	constexpr idx train_set = 12;
//	std::string parent = "Denoiser/";
//
//	//Input data
//	std::string in_folder = "Input/";
//	Tenarr x(3, 480, 360, train_set);
//#pragma omp parallel for
//	for(idx i = 0; i < train_set; i++)
//		x.chip(i, 3) = Image(parent + in_folder + std::to_string(i), 3).tensor_rgb();
//	//Output data
//	std::string out_folder = "Reference/";
//	std::string delta = "Delta/";
//	Tenarr y(3, 480, 360, train_set);
//#pragma omp parallel for
//	for(idx i = 0; i < train_set; i++) {
//		y.chip(i, 3) = Image(parent + out_folder + std::to_string(i), 3).tensor_rgb();// -x.chip(i, 3);
//		//Image(y.chip(i, 3).abs()).save(parent + delta + "/" + std::to_string(i) + ".png");
//	}
//	std::string netname = parent + "c5x32_c3x32x3_nr";
//#if 0
//	//hidden layers
//	vector<Layer*> top;
//	top.push_back(new Input(shp3(3, 480, 360)));
//	top.push_back(new Conv(32, 5, 1, 2, top.back(), true, Afun::t_lrelu));
//	top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_lrelu));
//	top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_lrelu));
//	top.push_back(new Conv(32, 3, 1, 1, top.back(), true, Afun::t_lrelu));
//	top.push_back(new Conv(3, 3, 1, 1, top.back(), true, Afun::t_lrelu));
//	top.push_back(new Output(top.back(), Efun::t_mae));
//	auto opt = new Adam(0.002f, Regular(RegulTag::L2, 0.1f));
//	NNet net(top, opt, netname);
//#else
//	NNet net(netname);
//#endif
//	constexpr idx rep = 20;
//	constexpr idx epoch = 100;
//	for(idx i = 0; i < rep; i++) {
//		if(!net.Train_Minibatch(x, y, epoch, lerp(0.001,0.001, float(i) / rep), 0.0, 12, epoch, 12, 10000)) break;
//		//if(!net.Train_parallel(x, y, 10, 0.002, -1, 100, train_set, 10, true)) break;
//		net.Save();
//#pragma omp parallel for
//		for(idx j = 0; j < train_set; j++) {
//			//if(j < train_set / 2)
//				Image(net.Compute(x.chip(j, 3)).abs()).save(netname + "/" + std::to_string(j) + ".png");
//			//else
//			//	Image(net.Compute(x.chip(j, 3)) + x.chip(j, 3)).save(netname + "/" + std::to_string(j) + ".png");
//		}
//	}
#endif
}
