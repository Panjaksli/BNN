#include "pch.h"
using namespace BNN;
// Tenarr(C,W,H,N) - (col major, channels are continuous)
#if 1
int main() {

#if 0
	//Inference code
	//NNet net("Upscaling_net");
	//if(!net.Valid()) return 1;
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
			out = resize(out, 1.5, 1.5, BNN::Cubic);
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
	std::string netname = parent + "c5x64_c3x64x3_4d2x3"; //c5x32_c3x32x3_2x3
	NNet upscl(parent + "c3_2x");
	Tenarr x(3, 240, 160, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		x.chip(i, 3) = Image(parent + in_folder + std::to_string(i), 3).tensor_rgb();
	//Output data
	std::string out_folder = "Reference/";
	Tenarr y(3, 480, 320, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		y.chip(i, 3) = Image(parent + out_folder + std::to_string(i), 3).tensor_rgb() - upscl.Compute(x.chip(i, 3));// -resize(x.chip(i, 3), 2, 2);
	//Test data
	std::string test_folder = "Test/";
	Tenarr z(3, 240, 160, test_set);
#pragma omp parallel for
	for(idx i = 0; i < test_set; i++)
		z.chip(i, 3) = Image(parent + test_folder + std::to_string(i), 3).tensor_rgb();
	/*for(idx i = 0; i < test_set; i++)
	Image(10.f * (Image(parent + "Test_ref/" + std::to_string(i), 3).tensor_rgb() - resize(z.chip(i, 3), 2, 2) ).abs()).save(parent + "Test_dy/" + std::to_string(i) + ".png");*/
#if 0
	//hidden layers
	vector<Layer*> top;
	top.push_back(new Input(shp3(3, 240, 160)));
	top.push_back(new Conv(64, 5, 1, 2, top.back(), true, Afun::t_lrelu));
	top.push_back(new Conv(64, 3, 1, 1, top.back(), true, Afun::t_lrelu));
	top.push_back(new Conv(64, 3, 1, 1, top.back(), true, Afun::t_lrelu));
	top.push_back(new Conv(64, 3, 1, 1, top.back(), true, Afun::t_lrelu));
	top.push_back(new Conv(3 * 4 * 4, 3, 1, 1, top.back(), true, Afun::t_lrelu));
	/*top.push_back(new SConv(3, 1, 1, top.back()));
	top.push_back(new Conv(32, 1, 1, 0, top.back(), true, Afun::t_cubl));*/
	top.push_back(new PixShuf(top.back(), 4));
	top.push_back(new Conv(3, 2, 2, 0, top.back(), true, Afun::t_lin));
	top.push_back(new Output(top.back(), Efun::t_mae));

	//top.push_back(new OutShuf(top.back(), 2, Efun::t_mae));
	auto opt = new Adam(0.002f);

	NNet net(top, opt, netname);
#else
	NNet net(netname, true);
#endif
	// * std::min(i / 4 + 1, 5)

	for(idx i = 0; i < 20; i++) {
		if(!net.Train_Minibatch(x, y, 20, 64, 16, -1, decay_rate(0.001f, i, 20))) break;
		//if(!net.Train_single(x, y, 100, 0.001, 16 * std::min(i / 4 + 1, 5), 100)) break;
		net.Save();
#pragma omp parallel for
		for(idx j = 0; j < test_set; j++) {
			/*if(j < 10)
				Image(net.Compute(z.chip(j, 3)).abs() * 10.f).save(netname + "/" + std::to_string(j) + ".png");
			else*/
			Image(net.Compute(z.chip(j, 3)) + upscl.Compute(z.chip(j, 3))).save(netname + "/" + std::to_string(j) + ".png");
			//Image(net.Compute(z.chip(j, 3))).save(netname + "/" + std::to_string(j) + ".png");
			//if(j < 10)
			//	Image(net.Compute(z.chip(j, 3)).abs() * 10.f).save(netname + "/" + std::to_string(j) + ".png");
			//else
			//	Image(net.Compute(z.chip(j, 3)) + resize(z.chip(j, 3), 2, 2)).save(netname + "/" + std::to_string(j) + ".png");
		}
		//for(idx j = 0; j < test_set; j++)
		//	Image(0.5f * (net.Compute(z.chip(j, 3)) + 1.f) /* + resize(z.chip(j, 3), 2, 2)*/).save(netname + "/" + std::to_string(j) + ".png");
		// + bil_ups(z.chip(j, 3), 2, 2)

		//net.Save_images(z);
	}
#endif

}
#else


int main() {
	int sz = 8;
	int ks = 5;
	int stride = 1;
	int nch = 2;
	Tensor x(nch, sz, sz);
	Tensor y(nch, sz / stride, sz / stride);
	Tensor w(nch * nch, ks, ks);
	x.setRandom();
	w.setRandom();
	Tenarr w1 = w.reshape(dim1<4>{nch, nch, ks, ks}).shuffle(dim1<4>{1, 0, 2, 3});
	double t = timer();
	y = Eigen::SpatialConvolution(x, w1, 1, 1, Eigen::PADDING_SAME, 1, 1, Eigen::NoOpOutputKernel());
	//::print(timer(t), y.dimension(0), y.dimension(1), y.dimension(2));
	print(timer(t), fsca(y.sum()).coeff(), "|");
	printnp(y);

	t = timer();
	convolve(y, x, w, stride, 2);
	print(timer(t), fsca(y.sum()).coeff(), "|");
	printnp(y);
	println("");
	/*for(int nch = 1; nch <= 128; nch *= 2) {
		Tensor x(nch, sz, sz);
		Tensor y(nch, sz / stride, sz / stride);
		Tensor w(nch * nch, ks, ks);
		x.setRandom();
		w.setRandom();
		Tenarr w1 = w.reshape(dim1<4>{nch, nch, ks, ks}).shuffle(dim1<4>{1, 0, 2, 3});
		double t = timer();
		conv2d1(y, x, w1, stride, 1, Eigen::PADDING_SAME);
		//::print(timer(t), y.dimension(0), y.dimension(1), y.dimension(2));
		print(timer(t), fsca(y.sum()).coeff(), "|");
		Tenarr w2 = w.reshape(dim1<4>{nch, nch, ks, ks}).shuffle(dim1<4>{0, 2, 3, 1});
		t = timer();
		conv2d2(y, x, w2, stride, 1, Eigen::PADDING_SAME);
		print(timer(t), fsca(y.sum()).coeff(), "|");
		t = timer();
		convolve(y, x, w, stride, 2);
		print(timer(t), fsca(y.sum()).coeff(), "|");
		println("");
	}*/
}
#endif
