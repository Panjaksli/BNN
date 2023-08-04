#include "Layer.h"
#include "Optimizer.h"
#include "NNet.h"
using namespace BNN;
int main() {
	constexpr int N = 100;
	vector<Layer*> topology;
	auto inp = new Input({ 3, 4, 4 });
	topology.push_back(new Conv(8, 3, 1, 1, inp));
	topology.push_back(new Conv(8, 2, 2, 0, topology.back()));
	topology.push_back(new TConv(8, 2, 2, 0, topology.back()));
	topology.push_back(new TConv(3, 3, 1, 1, topology.back()));
	auto outp = new Output({ 3,4,4 }, topology.back(), Afun::t_lin, Efun::t_mse);
	auto opt = new Adam(0.0002f, N, outp);
	NNet net(inp, topology, outp, opt);
	Tenarr x0(N, 3, 4, 4);
	Tenarr y0(N, 1, 2, 2);
	x0.setRandom();
	y0.setRandom();
	net.train(x0, x0, 0.01f, 10000);
}
