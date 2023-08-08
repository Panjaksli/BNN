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
		NNet(const std::string& name) : name(name) { Load(); }
		NNet(NNet first, const NNet& second) : NNet(first.Append(second)) {}
		NNet(const vector<Layer*>& graph, Optimizer* opt, const std::string& name = "Net") : graph(graph), optimizer(opt), name(name) { Compile(); }
		//fixed part of the network, learnable part
		bool Compile(bool log = 1);
		bool Train_single(const Tenarr& x0, const Tenarr& y0, int epochs = 1000, int nlog = 100);
		bool Train_parallel(const Tenarr& x0, const Tenarr& y0, int nthr = 16, int epochs = 1000, int nlog = 100);
		void Clear();
		void Print() const;
		dim1<3> In_dims()const { return graph.front()->idims(); }
		dim1<3> Out_dims()const { return graph.back()->odims(); }
		idx In_dim(idx i)const { return In_dims()[i]; }
		idx Out_dim(idx i)const { return Out_dims()[i]; }
		idx In_size()const { return product(In_dims()); }
		idx Out_size()const { return product(Out_dims()); }

		void Add_node(Layer* hidl) { if(hidl) { graph.push_back(hidl); compiled = false; } }
		void Add_node(Layer* hidl, idx id) { if(hidl && id < graph.size()) { graph.insert(graph.begin() + id, hidl); compiled = false; } }
		void Rem_node() { graph.pop_back(); compiled = false; }
		void Rem_node(idx id) { if(id < graph.size()) { graph.erase(graph.begin() + id); compiled = false; } }

		dim1<3> Dim_of(idx id)const { return graph[id]->odims(); }
		idx Size_of(idx id)const { return graph[id]->osize(); }
		
		Tensor Compute(const Tensor& x) const {
			return graph.front()->compute(x);
		}
		Tenarr Compute_batch(const Tenarr& x) const {
			Tenarr y(x.dimension(0), Out_dim(0), Out_dim(1), Out_dim(2));
			for(idx i = 0; i < x.dimension(0); i++) {
				y.chip(i, 0) = Compute(x.chip(i, 0));
			}
			return y;
		}
		NNet& Append(const NNet& other);
		void Save(const std::string &name) const;
		void Save() const;
		bool Load(const std::string &name);
		bool Load();
		void Save_image(const Tensor& x) const;
		void Save_images(const Tenarr& x) const;
	private:
		void init() {
			for(auto& g : graph) {
				g->init();
			}
			optimizer->reset_all();
		}
		void zero() {
			for(auto& g : graph) {
				g->zero();
			}
			optimizer->reset_all();
		}
		static bool valid_graph(const vector<Layer*> graph) {
			return graph.size() >= 2 && graph.front()->type() == t_Input && graph.back()->type() == t_Output;
		}
		bool valid() const { return optimizer && valid_graph(graph); }
		bool integrity_check(const dim1<4>& dim_x, const dim1<4>& dim_y) const;
		bool train_job(const Tenarr& x0, const Tenarr& y0, int epochs = 1000, int nlog = 100, bool log = 1);
		vector<Layer*> graph;
		Optimizer* optimizer = nullptr;
		std::string name = "Net";
		bool compiled = false;
	};
}