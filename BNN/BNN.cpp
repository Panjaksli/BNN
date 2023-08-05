#include "NNet/NNet.h"
#include "Image/Image.h"
using namespace BNN;
#if 1
int main() {
	Image in("in.png", 3);
	Image out("out.png", 3);
	vector<Layer*> topology;
	Input* inp = new Input(in.dim());
	topology.push_back(new TConv(3, 2, 2, 0, inp));
	topology.push_back(new TConv(3, 2, 2, 0, topology.back()));
	topology.push_back(new TConv(3, 2, 2, 0, topology.back()));
	//topology.back()->print();
	auto outp = new Output(out.dim(), topology.back(), Afun::t_lin, Efun::t_mse);
	//outp->print();
	auto opt = new Adam(0.001f, outp);
	NNet net(inp, topology, outp, opt, "Upscaler");
	Tenarr x(1, in.n, in.h, in.w);
	Tenarr y(1, out.n, out.h, out.w);
	x.chip(0, 0) = in.tensor();
	y.chip(0, 0) = out.tensor();
	for(int i = 0; i < 100; i++) {
		if(!net.Train_single(x, y, 1000, 100)) break;
		Image(net.Compute(x.chip(0, 0))).save("test.png");
	}
}
#else
int main() {
	constexpr int N = 800;
	vector<Layer*> topology;
	Input* inp = new Input({ 3, 4, 4 }); // 48
	topology.push_back(new Conv(8, 3, 1, 1, inp)); //64
	topology.push_back(new Conv(8, 2, 2, 0, topology.back())); //32
	topology.push_back(new Dense(16, topology.back())); //16
	topology.push_back(new Dropout(0.1f, topology.back()));
	topology.push_back(new Dense(32, topology.back())); //32
	topology.push_back(new TConv({8,2,2}, 8, 2, 2, 0, topology.back())); //64
	topology.push_back(new TConv(3, 3, 1, 1, topology.back())); //48
	auto outp = new Output({ 3,4,4 }, topology.back(), Afun::t_lin, Efun::t_mse);
	auto opt = new Adam(0.001f, outp);
	NNet net(inp, topology, outp, opt, "Autoencoder");
	Tenarr x0(N, 3, 4, 4);
	x0.setRandom();
	for(int i = 0; i < 100; i++) {
		if(!net.Train_parallel(x0, x0, 10, 100, 10)) break;
	}
}
#endif
