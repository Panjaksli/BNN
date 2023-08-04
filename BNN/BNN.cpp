#include "Layer.h"
#include "Optimizer.h"
#include "NNet.h"
using namespace BNN;
int main() {
	constexpr int N = 500;
	vector<Layer*> topology;
	auto inp = new Input({ 3, 4, 4 });
	topology.push_back(new Convol(8, 3, 1, 1, inp));
	topology.push_back(new Dense(16, topology.back()));
	topology.push_back(new Dense(18, topology.back()));
	topology.push_back(new Dense(4, topology.back()));
	auto outp = new Output({ 2,2 }, topology.back(), Afun::t_lin, Efun::t_mse);
	auto opt = new Adam(0.001f, N, outp);
	NNet net(inp, topology, outp, opt);
	Tenarr x0(N, 3, 4, 4);
	Tenarr y0(N, 1, 2, 2);
	x0.setRandom();
	y0.setRandom();
	net.train(x0, y0, 0.003f, 10000);
}
