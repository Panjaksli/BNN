#include <filesystem>
#include <fstream>
#include "NNet.h"
#include "../Image/Image.h"
using std::filesystem::path;
using std::filesystem::create_directories;
using std::filesystem::exists;

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
	NNet& NNet::operator=(NNet cpy) {
		Clear();
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
	NNet& NNet::Append(const NNet& other) {
		if(Out_size() == other.In_size()) {
			compiled = false;
			name = name + "_" + other.name;
			for(const auto& h : other.hidden) {
				hidden.push_back(h->clone());
			}
			if(output) delete output;
			output = other.output->clone();
			println("Message:  Joined network", name);
			Compile(1);
		}
		return *this;
	}
	bool NNet::integrity_check(const dim1<4>& dim_x, const dim1<4>& dim_y) const {
		if(dim_x[0] != dim_y[0]) {
			println("Error:  Number of training samples mismatch !");
			return false;
		}
		else if(!valid()) {
			println("Error:  Invalid network graph !");
			return false;
		}
		else if(dim_x[1] * dim_x[2] * dim_x[3] != input->isize()) {
			println("Error:  Input data mismatch !");
			return false;
		}
		else if(dim_y[1] * dim_y[2] * dim_y[3] != output->osize()) {
			println("Error:  Output data mismatch !");
			return false;
		}
		else if(!compiled) {
			println("Error:  Network was not compiled !");
			return false;
		}
		return true;
	}
	bool NNet::Compile(bool log) {
		if(compiled) return true;
		else if(!valid()) {
			if(log) println("Error:  Incomplete model ! (Node is missing...)");
			return compiled = false;
		}
		else {
			input->compile(nullptr, hidden[0]);
			for(idx i = 0; i < hidden.size(); i++) {
				if(!hidden[i]->compile(i == 0 ? input : hidden[i - 1], i < hidden.size() - 1 ? hidden[i + 1] : output)) {
					if(log)println("Error:  Dimension mismatch in node: ", i, "!", "Prev node:", hidden[i]->psize(), "This node:", hidden[i]->isize());
					return compiled = false;
				}
			}
		}
		if(!output->compile(hidden.back(), nullptr)) {
			if(log)println("Error:  Dimension mismatch in output node !", "Prev node:", output->psize(), "This node:", output->isize());
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
			bool res = nets[i].train_job(x0.slice(o, dx), y0.slice(o, dy), epochs, nlog, i == 0);
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
		if(!integrity_check(x0.dimensions(), y0.dimensions())) return false;
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
			if(cost > 1e3f) { println("Error:  Failed training !!!"); return false; }
			min_cost = min(cost, min_cost);
			optimizer->update_grad();
			if(log && i % log_step == 0) {
				printr("Epoch:", i, "Cost:", cost, "Min Cost:", min_cost, "Step:", timer(dt) / log_step, "Time:", timer(t));
				dt = timer();
			}
		}
		if(log) println("Message:  Trained Network:", name, "Min Cost:", min_cost, "Total Time:", timer(t));
		return true;
	}
	void NNet::Save(const std::string& folder) const {
		create_directories(folder);
		std::ofstream out(name + "/data.bin",std::ios::binary | std::ios::out);
		input->save(out);
		optimizer->save(out);
	}
	void NNet::Save() const { Save(name); }
	bool NNet::Load(const std::string& folder) {
		if(!exists(folder)) {
			println("Network file not found!");
			return false;
		}
		std::ifstream in(folder + "/data.bin", std::ios::binary | std::ios::in);
		Clear();
		while(in) {
			std::string token;
			in >> token;
			if(token == "Input")
				input = Input::load(in);
			else if(token == "Output")
				output = Output::load(in);
			else if(token == "Hidden")
				hidden.push_back(Hidden_load(in));
			else if(token == "Optimizer")
				optimizer = Optimizer_load(in);
			else break;
		}
		name = folder;
		println("Message:  Loaded Network:", name);
		Compile(1);
		return true;
	}
	bool NNet::Load() { return Load(name); }
	void NNet::Print() const {
		println("------------------------------------------------------------------------------------------------");
		std::cout << "Network\t|\t"; println(name);
		if(input)input->print();
		for(const auto& h : hidden) h->print();
		if(output)output->print();
		std::cout << ("Optimiz\t|\t"); if(optimizer) { optimizer->print(); }
		println("------------------------------------------------------------------------------------------------");
	}
	void NNet::Save_image(const Tensor& x) const {
		create_directories(name);
		Tensor y = Compute(x);
		Image(y).save(name + "/0.png");
	}
	void NNet::Save_images(const Tenarr& x) const {
		create_directories(name);
		Tenarr y = Compute_batch(x);
		for(int i = 0; i < y.dimension(0); i++)
			Image(y.chip(i, 0)).save(name + "/" + std::to_string(i + 1) + ".png");
	}
	void NNet::Clear() {
		compiled = 0;
		if(input)delete input;
		if(output)delete output;
		if(optimizer)delete optimizer;
		for(auto& h : hidden) delete h;
		hidden.clear();
	}
}