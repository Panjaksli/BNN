#include "NNet/NNet.h"
using namespace BNN;

int main() {
	constexpr int N = 800;
	vector<Layer*> topology;
	Input* inp = new Input({ 3, 4, 4 });
	topology.push_back(new Conv(8, 3, 1, 1, inp));
	topology.push_back(new Conv(8, 2, 2, 0, topology.back()));
	topology.push_back(new Dense(32, topology.back()));
	topology.push_back(new Dropout(0.1f, topology.back()));
	topology.push_back(new TConv({8,2,2}, 8, 2, 2, 0, topology.back()));
	topology.push_back(new TConv(3, 3, 1, 1, topology.back()));
	auto outp = new Output({ 3,4,4 }, topology.back(), Afun::t_lin, Efun::t_mse);
	auto opt = new Adam(0.001f, outp);
	NNet net(inp, topology, outp, opt, "Autoencoder");
	Tenarr x0(N, 3, 4, 4);
	x0.setRandom();
	for(int i = 0; i < 100; i++) {
		if(!net.Train_parallel(x0, x0, 16, 1000, 10)) break;
	}
}
