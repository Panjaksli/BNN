#pragma once
#include "Optimizer.h"
#include "../Layers/Layers.h"
namespace BNN {
	class NNet {
	public:
		NNet() {}
		~NNet() { Clear(); }
		NNet(const NNet& cpy);
		NNet& operator=(NNet cpy);
		NNet(const std::string& name, bool log = 0) : name(name) { Load(log); }
		NNet(NNet first, const NNet& second) : NNet(first.Append(second)) {}
		NNet(const vector<Layer*>& graph, Optimizer* opt, const std::string& name = "Net") : graph(graph), optimizer(opt), name(name) { Compile(); }
		//fixed part of the network, learnable part
		bool Compile(bool log = 1);
		bool Train_single(const Tenarr& x0, const Tenarr& y0, idx epochs = 100, float rate = -1, idx batch = -1, idx nlog = -1);
		bool Train_parallel(const Tenarr& x0, const Tenarr& y0, idx epochs = 100, float rate = -1, idx batch = -1, idx nlog = -1, idx threads = 16, idx steps = -1, bool keep_grad = 0);
		bool Train_Minibatch(const Tenarr& x0, const Tenarr& y0, idx epochs = 100, idx batch_sz = -1, idx threads = -1, idx steps = -1, float rate = -1);
		void Clear();
		void Print() const;
		dim1<3> In_dims()const { return graph.front()->idims(); }
		dim1<3> Out_dims()const { return graph.back()->odims(); }
		idx In_dim(idx i)const { return In_dims()[i]; }
		idx Out_dim(idx i)const { return Out_dims()[i]; }
		idx In_size()const { return graph.front()->isize(); }
		idx Out_size()const { return graph.back()->osize(); }
		Optimizer* Optim() { return optimizer; }
		const Optimizer* Optim() const { return optimizer; }
		void Set_optim(Optimizer* opt) { if(opt) { if(optimizer) delete optimizer; optimizer = opt; compiled = 0; } }
		void Add_node(Layer* hidl) { if(hidl) { graph.push_back(hidl); compiled = false; } }
		void Add_node(Layer* hidl, idx id) { if(hidl && id < graph.size()) { graph.insert(graph.begin() + id, hidl); compiled = false; } }
		void Rem_node() { if(graph.size() > 0) { delete graph.back(); graph.pop_back(); compiled = false; } }
		void Rem_node(idx id) { if(id < graph.size()) { delete(graph[id]); graph.erase(graph.begin() + id); compiled = false; } }

		dim1<3> Dim_of(idx id)const { return graph[id]->odims(); }
		idx Size_of(idx id)const { return graph[id]->osize(); }

		Tensor Compute(const Tensor& x) const {
			return graph.front()->compute(x);
		}
		Tenarr Compute_batch(const Tenarr& x) const {
			Tenarr y(Out_dim(0), Out_dim(1), Out_dim(2), x.dimension(3));
			for(idx i = 0; i < x.dimension(3); i++) {
				y.chip(i, 3) = Compute(x.chip(i, 3));
			}
			return y;
		}
		//compute but agnostic to the input size (mostly), dont use with dense layers!!!!!!!!!!!!!!!!
		Tensor Compute_DS(const Tensor& x) const {
			return graph.front()->compute_ds(x);
		}
		NNet Clone_raw() const {
			NNet cpy;
			cpy.name = name;
			cpy.compiled = false;
			for(const auto& g : graph) {
				cpy.graph.push_back(g->clone());
			}
			cpy.optimizer = new SGD(Optim()->alpha, Optim()->reg);
			cpy.optimizer->reset_all();
			cpy.Compile(0);
			return cpy;
		}
		NNet& Append(const NNet& other);
		Layer* Back() { return graph.back(); }
		Layer* Front() { return graph.front(); }
		void Save(const std::string& name) const;
		void Save() const;
		bool Load(const std::string& name, bool log);
		bool Load(bool log);
		void Save_image(const Tensor& x) const;
		void Save_image_DS(const Tensor& x) const;
		void Save_images(const Tenarr& x) const;
		void Init() {
			for(auto& g : graph) {
				g->init();
			}
			optimizer->reset_all();
		}
		bool Valid() const { return optimizer && valid_graph(graph); }
	private:
		void Zero() {
			for(auto& g : graph) {
				g->zero();
			}
			optimizer->reset_all();
		}
		static bool valid_graph(const vector<Layer*> graph) {
			return graph.size() >= 2 && graph.front()->type() == t_Input && (graph.back()->type() == t_Output || graph.back()->type() == t_OutShuf);
		}
		bool integrity_check(const dim1<4>& dim_x, const dim1<4>& dim_y) const;
		float train_job(const Tenarr& x0, const Tenarr& y0, const vector<int>& indices, idx beg, idx end,
			idx epochs = 1000, idx nlog = 100, idx index = 0, idx ep_off = 0, bool log = 1);
		vector<Layer*> graph;
		Optimizer* optimizer = nullptr;
		std::string name = "Net";
		float best_cost = 1e6f;
		bool compiled = false;
	};
}