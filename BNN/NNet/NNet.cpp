#include "NNet.h"
namespace BNN {
	NNet::NNet(const NNet& cpy) {
		name = cpy.name;
		compiled = false;
		input = cpy.input->clone();
		for(const auto& h : cpy.hidden) {
			hidden.push_back(h->clone());
		}
		output = cpy.output->clone();
		optimizer = cpy.optimizer->clone();
		Compile(0);
	}
	NNet& NNet::operator=(const NNet& cpy) {
		Clear();
		compiled = false;
		name = cpy.name;
		input = cpy.input->clone();
		for(const auto& h : cpy.hidden) {
			hidden.push_back(h->clone());
		}
		output = cpy.output->clone();
		optimizer = cpy.optimizer->clone();
		Compile(0);
		return *this;
	}
	bool NNet::integrity_check(const dim1<4> &dim_x, const dim1<4> &dim_y) const {
		if(dim_x[0] != dim_y[0]) {
			println("Number of training samples mismatch !");
			return false;
		}
		else if(!valid()) {
			println("Invalid network graph !");
			return false;
		}
		else if(dim_x[1] * dim_x[2] * dim_x[3] != product(input->dim_out())) {
			println("Input data mismatch !");
			return false;
		}
		else if(dim_y[1] * dim_y[2] * dim_y[3] != product(output->dim_out())) {
			println("Output data mismatch !");
			return false;
		}
		else if(!compiled) {
			println("Network was not compiled !");
			return false;
		}
		return true;
	}
	bool NNet::Compile(bool log) {
		if(compiled) return true;
		else if(!valid()) {
			if(log)println("Error:  Incomplete model ! (Node is missing...)");
			return compiled = false;
		}
		else {
			input->compile(nullptr, hidden[0]);
			for(idx i = 0; i < hidden.size(); i++) {
				if(!hidden[i]->compile(i == 0 ? input : hidden[i - 1], i < hidden.size() - 1 ? hidden[i + 1] : output)) {
					if(log)println("Error:  Dimension mismatch in node: ", i, "!", "Prev nodes:", hidden[i]->sz_x(), "This nodes:", hidden[i]->prev->sz_in());
					return compiled = false;
				}
			}
		}
		if(!output->compile(hidden.back(), nullptr)) {
			if(log)println("Error:  Dimension mismatch in output node !");
			return compiled = false;
		}
		optimizer->compile(output);
		if(log)println("Message:  Network compiled succesfully:");
		if(log)Print();
		return compiled = true;
	}

	bool NNet::Train_parallel(const Tenarr& x0, const Tenarr& y0, int nthr, int epochs, int nlog) {
		if(!integrity_check(x0.dimensions(), y0.dimensions())) return false;
		if(x0.dimension(0) < nthr) nthr = x0.dimension(0);
		float mult = 1.f / nthr;
		int step = x0.dimension(0) / nthr;
		bool result = 1;
		vector<NNet> nets(nthr, *this);
		dim1<4> dx{step, x0.dimension(1), x0.dimension(2), x0.dimension(3)};
		dim1<4> dy{step, y0.dimension(1), y0.dimension(2), y0.dimension(3)};
#pragma omp parallel for
		for(int i = 0; i < nthr; i++) {
			dim1<4> o{i* step, 0, 0, 0};
			bool res = nets[i].train_job(x0.slice(o,dx), y0.slice(o,dy), epochs, nlog, i == 0);
#pragma omp atomic
			result &= res;
		}
		if(!result) return false;
		NNet net(*this);
		net.zero();
		for(const auto& n : nets) {
			for(int i = 0; i < net.hidden.size(); i++) {
				if(net.hidden[i]->get_b()) *net.hidden[i]->get_b() += mult * *n.hidden[i]->get_b();
				if(net.hidden[i]->get_w()) *net.hidden[i]->get_w() += mult * *n.hidden[i]->get_w();
			}
		}
		*this = net;
		return true;
	}
	bool NNet::Train_single(const Tenarr& x0, const Tenarr& y0, int epochs, int nlog) {
		if(!integrity_check(x0.dimensions(),y0.dimensions())) return false;
		return train_job(x0, y0, epochs, nlog);
	}

	bool NNet::train_job(const Tenarr& x0, const Tenarr& y0, int epochs, int nlog, bool log) {
		optimizer->inv_n = 1.f / x0.dimension(0);
		optimizer->reset_all();
		float min_cost = 1e6f;
		double t = timer();
		double dt = timer();
		float inv_ep = 1.f / epochs;
		int log_step = epochs / nlog;
		log_step = max(1, log_step);
		for(int i = 1; i <= epochs; i++) {
			float cost = 0;
			for(int j = 0; j < x0.dimension(0); j++) {
				input->predict(x0.chip(j, 0));
				cost += output->error(y0.chip(j, 0)) * inv_ep;
				optimizer->get_grad();
			}
			if(cost > 1e3f) { println("Failed training !!!"); return false;}
			min_cost = min(cost, min_cost);
			optimizer->update_grad();
			if(log && i % log_step == 0) {
				printr("Epoch:", i, "Cost:", cost, "Min Cost:", min_cost,  "Step:", timer(dt) / log_step, "Time:", timer(t));
				dt = timer();
			}
		}
		if(log) println("Trained Network:", name, "Min Cost:", min_cost, "Total Time:", timer(t));
		//println("Trained Network:", name,"Thread:", omp_get_thread_num(), "Min Cost:", min_cost, "Total Time:", timer(t));
		return true;
	}

	void NNet::Print() const {
		println("------------------------------------------------------------------------------------------------");
		std::cout << "Network\t|\t"; println(name);
		if(input)input->print();
		for(const auto& h : hidden) h->print();
		if(output)output->print();
		std::cout << ("Optimiz\t|\t"); if(optimizer) { optimizer->print(); }
		println("------------------------------------------------------------------------------------------------");
	}

	void NNet::Clear() {
		if(input)delete input;
		if(output)delete output;
		if(optimizer)delete optimizer;
		for(auto& h : hidden) delete h;
		hidden.clear();
	}
}