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
		if(dim_x[3] != dim_y[3]) {
			println("Error:  Number of training samples mismatch !");
			return false;
		}
		else if(!Valid()) {
			println("Error:  Invalid network graph !");
			return false;
		}
		else if(dim_x[0] * dim_x[1] * dim_x[2] != In_size()) {
			println("Error:  Input data mismatch !");
			return false;
		}
		else if(dim_y[0] * dim_y[1] * dim_y[2] != Out_size()) {
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

	bool NNet::Train_parallel(const Tenarr& x0, const Tenarr& y0, idx epochs, float rate, idx batch, idx nlog, idx threads, idx steps, bool keep_grad) {
		if(!integrity_check(x0.dimensions(), y0.dimensions())) return false;
		if(rate > 0) optimizer->alpha = rate;
		if(batch < 0 || batch > x0.dimension(3)) batch = x0.dimension(3);
		if(batch < threads) threads = batch;
		if(nlog <= 0) nlog = epochs;
		if(steps <= 0) steps = 1;
		bool result = 1;
		float cost = 0;
		float mult = 1.f / threads;
		idx batch_sz = batch / threads;
		printr("Training Network:", name, "Dataset:", batch, "Batches:", threads, "Batch size:", batch_sz);
		double t = timer();
		optimizer->reset_all();
		optimizer->inv_n = 1.f / batch_sz;
		NNet net(*this);
		for(idx step = 0; step < steps; step++) {
			cost = 0;
			vector<NNet> nets(threads, net);
			vector<int> indices = shuffled(x0.dimension(3));
#pragma omp parallel for shared(x0,y0,nets,indices)
			for(idx i = 0; i < threads; i++) {
				idx off = i * batch_sz;
				float cst = nets[i].train_job(x0, y0, indices, off, off + batch_sz, epochs, nlog, i, step * epochs, i == 0);
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
			cost *= mult;
			net.Zero();
			for(const auto& n : nets) {
				for(idx i = 1; i < net.graph.size() - 1; i++) {
					if(net.graph[i]->get_b()) *net.graph[i]->get_b() += mult * (*n.graph[i]->get_b());
					if(net.graph[i]->get_w()) *net.graph[i]->get_w() += mult * (*n.graph[i]->get_w());
				}
				if(keep_grad) {
					for(idx i = 0; i < net.optimizer->size(); i++) {
						if(net.optimizer->get_vw(i)) *net.optimizer->get_vw(i) += mult * (*n.optimizer->get_vw(i));
						if(net.optimizer->get_mw(i)) *net.optimizer->get_mw(i) += mult * (*n.optimizer->get_mw(i));
						if(net.optimizer->get_vb(i)) *net.optimizer->get_vb(i) += mult * (*n.optimizer->get_vb(i));
						if(net.optimizer->get_mb(i)) *net.optimizer->get_mb(i) += mult * (*n.optimizer->get_mb(i));
					}
				}
			}
			best_cost = fminf(cost, best_cost);
			printr("Step:", step + 1, "Cost:", cost, "Time:", timer(t), "                    ");
		}
		println("Trained Network:", name, "Cost:", cost, "Best:", best_cost, "Time:", timer(t), "                ");
		*this = net;
		return true;
	}
	bool NNet::Train_single(const Tenarr& x0, const Tenarr& y0, idx epochs, float rate, idx batch, idx nlog) {
		if(!integrity_check(x0.dimensions(), y0.dimensions())) return false;
		if(rate > 0) optimizer->alpha = rate;
		if(batch < 0 || batch > x0.dimension(3)) batch = x0.dimension(3);
		if(nlog < 0) nlog = epochs;
		optimizer->inv_n = 1.f / batch;
		optimizer->reset_all();
		double t = timer();
		vector<int> indices = shuffled(x0.dimension(3));
		float cost = train_job(x0, y0, indices, 0, batch, epochs, nlog);
		if(cost < 0) {
			println("Failed training the network !!!");
			return false;
		}
		best_cost = fminf(cost, best_cost);
		println("Trained Network:", name, "Cost:", cost, "Best:", best_cost, "Time:", timer(t), "           ");
		return true;
	}

	float NNet::train_job(const Tenarr& x0, const Tenarr& y0, const vector<int>& indices, idx beg, idx end, idx epochs, idx nlog, idx index, idx ep_off, bool log) {
		float min_cost = 1e6f;
		double t = timer();
		double dt = timer();
		idx log_step = epochs / nlog;
		idx max_epochs = 3 * epochs / 2;
		log_step = max(1, log_step);
		float cost = 0;
		for(idx i = 1; i <= epochs; i++) {
			cost = 0;
			for(idx j = beg; j < end; j++) {
				graph.front()->predict(x0.chip(indices[j], 3));
				cost += graph.back()->error(y0.chip(indices[j], 3));
				optimizer->get_grad();
			}
			cost *= optimizer->inv_n;
			if(cost > 1e6f) { return -1; }
			min_cost = fminf(cost, min_cost);
			optimizer->update_grad();
			if(log && i % log_step == 0) {
				printr("ID:", index, "Epoch:", i + ep_off, "Cost:", cost, "Min:", min_cost, "Step:", timer(dt) / log_step, "Time:", timer(t), "           ");
				dt = timer();
			}
			if(i >= max_epochs) break;
			else if(i == epochs && cost > min_cost) epochs++;
		}
		return cost;
	}
	bool NNet::Train_Minibatch(const Tenarr& x0, const Tenarr& y0, idx epochs, idx batch_sz, idx threads, idx steps, float rate) {
		idx dataset_sz = x0.dimension(3);
		if(!integrity_check(x0.dimensions(), y0.dimensions()) || dataset_sz <= 0) return false;
		if(batch_sz < 0 || batch_sz > dataset_sz) batch_sz = dataset_sz;
		if(batch_sz < threads) threads = batch_sz;
		if(steps <= 0 || steps >= dataset_sz / batch_sz) steps = dataset_sz / batch_sz;
		printr("Training Network:", name, "Data:", dataset_sz, "Steps:", steps, "Batch:", batch_sz, "Threads:", threads);
		idx thr_work = batch_sz / threads;
		float cost = 0;
		double t = timer();
		NNet net(*this);
		//Creates copy of this NNet, zeroes all gradients and buffers in optimizer
		net.optimizer->reset_all();
		//Divisor for gradients, is equal to 1 / minibatch
		net.optimizer->inv_n = 1.f / batch_sz;
		if(rate > 0) net.optimizer->alpha = rate;
		// Pool of NNets, each has SGD optimizer containing ONLY gradients dw and db, those are INITIALIZED TO ZERO
		vector<NNet> nets(threads, net.Clone_raw());
		// Shuffled array of indices of dataset size 
		vector<int> indices = shuffled(dataset_sz);
		for(idx epoch = 0; epoch < epochs; epoch++) {
			double dt = timer();
			cost = 0;
			for(idx step = 0; step < steps; step++) {
				//For each NNet (thread) accumulate gradients for N elements, N = minibatch / threads
#pragma omp parallel for shared(x0,y0,nets,indices) reduction(+:cost) schedule(static, 1)
				for(idx th = 0; th < threads; th++) {
					float thr_cost = 0;
					idx off = step * batch_sz + th * thr_work;
					for(idx j = 0; j < thr_work; j++) {
						nets[th].graph.front()->predict(x0.chip(indices[off + j], 3));
						thr_cost += nets[th].graph.back()->error(y0.chip(indices[off + j], 3));
						nets[th].optimizer->get_grad();
					}
					cost += thr_cost;
				}
				//Compute current cost as acc_cost / minibatch
				if(cost / ((step + 1) * batch_sz) > 1e6) return false;
				// Accumulate gradients from each thread to a single net
				for(idx th = 0; th < threads; th++) {
					for(idx i = 0; i < net.optimizer->size(); i++) {
						if(net.optimizer->get_dw(i))
							*net.optimizer->get_dw(i) += (*nets[th].optimizer->get_dw(i));
						if(net.optimizer->get_db(i))
							*net.optimizer->get_db(i) += (*nets[th].optimizer->get_db(i));
					}
					// Zero gradients dw and db in each threads
					nets[th].Optim()->reset_grad();
				}
				// Update the main net weights, resets gradients but not velocity etc.
				net.optimizer->update_grad();
				// Copy new weights to all threads
#pragma omp parallel for shared(nets, net) schedule(static, 1)
				for(idx th = 0; th < threads; th++) {
					for(idx node = 0; node < net.graph.size(); node++) {
						if(nets[th].graph[node]->get_w()) *nets[th].graph[node]->get_w() = *net.graph[node]->get_w();
						if(nets[th].graph[node]->get_b()) *nets[th].graph[node]->get_b() = *net.graph[node]->get_b();
					}
				}
			}
			cost /= steps * batch_sz;
			best_cost = fminf(cost, best_cost);
			// Shuffle dataset indices
			shuffle(indices);
			dt = timer(dt);
			printr("Ep:", epoch + 1, "Loss:", cost, "Best:", best_cost, "Epoch [s]:", dt, "Step [ms]:", 1000 * dt / steps, "Time [s]:", timer(t), "           ");
		}
		println("Trained Network:", name, "Cost:", cost, "Best:", best_cost, "Time:", timer(t), "                ");
		std::swap(*this, net);
		return true;
	}
	void NNet::Save(const std::string& folder) const {
		create_directories(folder);
		std::ofstream out(name + "/data.bin", std::ios::binary | std::ios::out);
		graph.front()->save(out);
		optimizer->save(out);
		out << "Cost" SPC best_cost << "\n";
	}
	void NNet::Save() const { Save(name); }
	bool NNet::Load(const std::string& folder, bool log) {
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
			else if(token == "Cost")
				in >> best_cost;
			else break;
		}
		name = folder;
		println("Message:  Loaded Network:", name);
		Compile(log);
		return true;
	}
	bool NNet::Load(bool log) { return Load(name, log); }
	void NNet::Print() const {
		println("------------------------------------------------------------------------------------------------");
		std::cout << "Network\t|\t"; println(name, "\tCost:", best_cost);
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
		for(idx i = 0; i < y.dimension(3); i++)
			Image(y.chip(i, 3)).save(name + "/" + std::to_string(i) + ".png");
	}
	void NNet::Clear() {
		compiled = false;
		if(optimizer)delete optimizer;
		for(auto& h : graph) delete h;
		graph.clear();
	}
}

