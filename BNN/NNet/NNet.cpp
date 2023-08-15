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
		for(const auto& g : cpy.graph) {
			graph.push_back(g->clone());
		}
		optimizer = cpy.optimizer->clone();
		Compile(0);
	}
	NNet& NNet::operator=(NNet cpy) {
		Clear();
		name = cpy.name;
		for(const auto& g : cpy.graph) {
			graph.push_back(g->clone());
		}
		optimizer = cpy.optimizer->clone();
		Compile(0);
		return *this;
	}
	NNet& NNet::Append(const NNet& other) {
		if(Out_size() == other.In_size()) {
			compiled = false;
			name = name + "_" + other.name;
			Rem_node();
			for(idx i = 1; i < other.graph.size(); i++) {
				graph.push_back(other.graph[i]->clone());
			}
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
		else if(!Valid()) {
			println("Error:  Invalid network graph !");
			return false;
		}
		else if(dim_x[1] * dim_x[2] * dim_x[3] != In_size()) {
			println("Error:  Input data mismatch !");
			return false;
		}
		else if(dim_y[1] * dim_y[2] * dim_y[3] != Out_size()) {
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
		else if(!Valid()) {
			if(log) println("Error:  Incomplete model ! (Node is missing...)");
			return compiled = false;
		}
		else {
			for(idx i = 0; i < graph.size(); i++) {
				Layer* prev = i ? graph[i - 1] : nullptr;
				Layer* next = graph.size() - i - 1 ? graph[i + 1] : nullptr;
				if(!graph[i]->compile(prev, next)) {
					if(log)println("Error:  Dimension mismatch in node: ", i, "!", "Prev node:", graph[i]->psize(), "This node:", graph[i]->isize());
					return compiled = false;
				}
			}
		}
		optimizer->compile(graph.front());
		if(log)println("Message:  Network compiled succesfully:");
		if(log)Print();
		return compiled = true;
	}

	bool NNet::Train_parallel(const Tenarr& x0, const Tenarr& y0, idx nthr, float rate, idx epochs, idx nlog) {
		if(!integrity_check(x0.dimensions(), y0.dimensions())) return false;
		if(rate > 0) optimizer->alpha = rate;
		if(x0.dimension(0) < nthr) nthr = x0.dimension(0);
		double t = timer();
		bool result = 1;
		float cost = 0;
		float mult = 1.f / nthr;
		idx step = x0.dimension(0) / nthr;
		vector<NNet> nets(nthr, *this);
		dim1<4> dx{ step, x0.dimension(1), x0.dimension(2), x0.dimension(3) };
		dim1<4> dy{ step, y0.dimension(1), y0.dimension(2), y0.dimension(3) };
		vector<int> indices = shuffled(x0.dimension(0));
		Tenarr x(x0.dimensions());
		Tenarr y(y0.dimensions());
		for(idx i = 0; i < indices.size(); i++) {
			x.chip(i, 0) = x0.chip(indices[i], 0);
			y.chip(i, 0) = y0.chip(indices[i], 0);
		}
#pragma omp parallel for
		for(idx i = 0; i < nthr; i++) {
			dim1<4> o{ i * step, 0, 0, 0 };
			float cst = nets[i].train_job(x.slice(o, dx), y.slice(o, dy), epochs, nlog, i, i == 0);
			bool res = cst >= 0;
#pragma omp atomic
			result &= res;
#pragma omp atomic
			cost += cst;
		}
		if(!result) {
			println("Failed training the network !!!");
			return false;
		}
		NNet net(*this);
		net.Zero();
		for(const auto& n : nets) {
			for(idx i = 1; i < net.graph.size() - 1; i++) {
				if(net.graph[i]->get_b()) *net.graph[i]->get_b() += mult * (*n.graph[i]->get_b());
				if(net.graph[i]->get_w()) *net.graph[i]->get_w() += mult * (*n.graph[i]->get_w());
			}
		}
		println("Trained Network:", name, "Cost:", cost * mult, "Time:", timer(t));
		*this = net;
		return true;
	}
	bool NNet::Train_single(const Tenarr& x0, const Tenarr& y0, float rate, idx epochs, idx nlog) {
		if(!integrity_check(x0.dimensions(), y0.dimensions())) return false;
		if(rate > 0) optimizer->alpha = rate;
		double t = timer();
		float cost = train_job(x0, y0, epochs, nlog);
		if(cost < 0) {
			println("Failed training the network !!!");
			return false;
		}
			println("Trained Network:", name, "Cost:", cost, "Time:", timer(t));
			return true;
	}

	float NNet::train_job(const Tenarr& x0, const Tenarr& y0, idx epochs, idx nlog, idx index, bool log) {
		optimizer->inv_n = 1.f / x0.dimension(0);
		optimizer->reset_all();
		float min_cost = 1e6f;
		double t = timer();
		double dt = timer();
		idx log_step = epochs / nlog;
		idx max_epochs = 3 * epochs / 2;
		log_step = max(1, log_step);
		float cost = 0;
		for(idx i = 1; i <= epochs; i++) {
			cost = 0;
			for(idx j = 0; j < x0.dimension(0); j++) {
				graph.front()->predict(x0.chip(j, 0));
				cost += graph.back()->error(y0.chip(j, 0));
				optimizer->get_grad();
			}
			cost *= optimizer->inv_n;
			if(cost > 1e3f) { return -1; }
			min_cost = fminf(cost, min_cost);
			optimizer->update_grad();
			if(log && i % log_step == 0) {
				printr("ID:", index, "Epoch:", i, "Cost:", cost, "Min:", min_cost, "Step:", timer(dt) / log_step, "Time:", timer(t));
				dt = timer();
			}
			if(i >= max_epochs) break;
			else if(i == epochs && cost > min_cost) epochs++;
		}
		return cost;
	}
	void NNet::Save(const std::string& folder) const {
		create_directories(folder);
		std::ofstream out(name + "/data.bin", std::ios::binary | std::ios::out);
		graph.front()->save(out);
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
			if(token == "Hidden")
				graph.push_back(Layer_load(in));
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
		for(const auto& g : graph) g->print();
		std::cout << ("Optimiz\t|\t"); if(optimizer) { optimizer->print(); }
		println("------------------------------------------------------------------------------------------------");
	}
	void NNet::Save_image(const Tensor& x) const {
		create_directories(name);
		Tensor y = Compute(x);
		Image(y).save(name + "/0.png");
	}
	void NNet::Save_image_DS(const Tensor& x) const {
		create_directories(name);
		Tensor y = Compute_DS(x);
		Image(y).save(name + "/0.png");
	}
	void NNet::Save_images(const Tenarr& x) const {
		create_directories(name);
		Tenarr y = Compute_batch(x);
		for(idx i = 0; i < y.dimension(0); i++)
			Image(y.chip(i, 0)).save(name + "/" + std::to_string(i) + ".png");
	}
	void NNet::Clear() {
		compiled = false;
		if(optimizer)delete optimizer;
		for(auto& h : graph) delete h;
		graph.clear();
	}
}