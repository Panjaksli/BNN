#include "Layer.h"
using namespace BNN;

int main() {
	Layer* i = new Input(3, 4, 4);
	Layer* c = new Convol({3,4,4},{4,3,3},{1,1},{1,1},i);
	Layer* d1 = new Dense(4*4*4, c);
	Layer* d2 = new Dense(16, d1);
	Layer* d3 = new Dense(1, d2);
	Layer* o = new Output(1, d3);
	Tensor x(3, 4, 4);
	Tensor y(1, 1, 1);
	x.setRandom();
	y.setConstant(0.5f);
	for (int k = 0; k < 2000000; k++) {
		i->forward(x);
		o->backward(y);
		println(o->get_cost());
	}
	/*print_np(y);
	print_np(i->predict_batch(x,1,1,1));*/
	//println(timer(t));

}
