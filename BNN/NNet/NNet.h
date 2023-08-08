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
		NNet(Input* in, Output* out, Optimizer* opt, const std::string& name = "Net") : input(in), output(out), optimizer(opt) {}
		NNet(NNet first, const NNet& second) : NNet(first.Append(second)) {}
		NNet(Input* in, const vector<Layer*>& hid, Output* out, Optimizer* opt, const std::string& name = "Net") : input(in), output(out), hidden(hid), optimizer(opt), name(name) { Compile(); }
		bool Compile(bool log = 1);
		bool Train_single(const Tenarr& x0, const Tenarr& y0, int epochs = 1000, int nlog = 100);
		bool Train_parallel(const Tenarr& x0, const Tenarr& y0, int nthr = 16, int epochs = 1000, int nlog = 100);
		void Clear();
		void Print() const;
		dim1<3> In_dims()const { return input ? input->odims() : dim1<3>{ 0,0,0 }; }
		dim1<3> Out_dims()const { return output ? output->odims() : dim1<3>{ 0,0,0 }; }
		idx In_dim(idx i)const { return In_dims()[i]; }
		idx Out_dim(idx i)const { return Out_dims()[i]; }
		idx In_size()const { return input ? input->osize() : 0; };
		idx Out_size()const { return output ? output->osize() : 0; }
		void Set_input(Input* in) { if(in) { input = in; compiled = false; } }
		void Set_output(Output* out) { if(out) { output = out; compiled = false; } }
		void Add_hidden(Layer* hidl) { if(hidl) { hidden.push_back(hidl); compiled = false; } }
		void Add_hidden(Layer* hidl, idx id) { if(hidl && id < hidden.size()) { hidden.insert(hidden.begin() + id, hidl); compiled = false; } }
		dim1<3> Dim_of(idx id)const { return hidden[id]->odims(); }
		idx Size_of(idx id)const { return hidden[id]->osize(); }
		void Rem_hidden() { hidden.pop_back(); compiled = false; }
		void Rem_hidden(idx id) { if(id < hidden.size()) { hidden.erase(hidden.begin() + id); compiled = false; } }
		Tensor Compute(const Tensor& x) const {
			return input->compute(x);
		}
		Tenarr Compute_batch(const Tenarr& x) const {
			return input->compute_batch(x);
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
			for(auto& h : hidden) {
				h->init();
			}
			optimizer->reset_all();
		}
		void zero() {
			for(auto& h : hidden) {
				h->zero();
			}
			optimizer->reset_all();
		}
		bool valid() const { return input && output && optimizer && hidden.size() > 0; }
		bool integrity_check(const dim1<4>& dim_x, const dim1<4>& dim_y) const;
		bool train_job(const Tenarr& x0, const Tenarr& y0, int epochs = 1000, int nlog = 100, bool log = 1);
		Input* input = nullptr;
		Output* output = nullptr;
		vector<Layer*> hidden;
		Optimizer* optimizer = nullptr;
		std::string name = "Net";
		bool compiled = false;
	};
}