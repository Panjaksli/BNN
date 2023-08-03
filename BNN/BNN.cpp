#include "Layer.h"
#include "Optimizer.h"
#include "NNet.h"
using namespace BNN;

int main() {
	constexpr int N = 100;
	vector<Layer*> topology;
	auto inp = new Input({ 3, 4, 4 });
	topology.push_back(new Convol(8, 3, 1, 1, inp));
	topology.push_back(new Dense(16, topology.back()));
	topology.push_back(new Dense(8, topology.back()));
	topology.push_back(new Dense(4, topology.back()));
	auto outp = new Output({ 2,2 }, topology.back(), Afun::t_lin, Efun::t_mse);
	vector<Optimizer*> optimizer;
	optimizer.push_back(new Adam(0.001f, N, outp));
	NNet net(inp, topology, outp, optimizer[0]);;
	/*optimizer.push_back(new RMSprop(0.01f, N, outp));
	optimizer.push_back(new NAG(0.01f, N, outp));
	optimizer.push_back(new AGD(0.01f, N, outp));
	optimizer.push_back(new SGD(0.01f, N, outp));*/
	/*Tenarr x0(N, 3, 4, 4);
	Tenarr y0(N, 1, 2, 2);
	x0.setRandom();
	y0.setRandom();
	for (auto& opt : optimizer) {
		for (int i = 0; i < 20000; i++) {
			for (int j = 0; j < N; j++) {
				inp->predict(x0.chip(j, 0));
				outp->error(y0.chip(j, 0));
				opt->get_grad();
			}
			opt->update_grad();
		}
		float cost = 0;
		for (int j = 0; j < N; j++) {
			inp->predict(x0.chip(j, 0));
			cost += outp->error(y0.chip(j, 0));
		}
		println(cost / N);
	}*/
	//println(timer(t));

}
