#pragma once
#include "Optimizer.h"
#include "../Layers/Layers.h"
namespace BNN {
	class NNet {
	public:
		NNet() {}
		~NNet() { Clear(); }
		NNet(const NNet& cpy);
		NNet& operator=(const NNet& cpy);
		NNet(Input* in, Output* out, Optimizer* opt, const std::string& name = "Net") : input(in), output(out), optimizer(opt) {}
		NNet(Input* in, const vector<Layer*>& hid, Output* out, Optimizer* opt, const std::string& name = "Net") : input(in), output(out), hidden(hid), optimizer(opt), name(name) { Compile(); }
		bool Compile(bool log = 1);
		bool Train_single(const Tenarr& x0, const Tenarr& y0, int epochs = 1000, int nlog = 100);
		bool Train_parallel(const Tenarr& x0, const Tenarr& y0, int nthr = 16, int epochs = 1000, int nlog = 100);
		void Clear();
		void Print() const;
		void Set_input(Input* in) { if(in) { input = in; compiled = false; } }
		void Set_output(Output* out) { if(out) { output = out; compiled = false; } }
		void Add_hidden(Layer* hidl) { if(hidl) { hidden.push_back(hidl); compiled = false; } }
		void Rem_hidden() { hidden.pop_back(); compiled = false; }
		void Rem_hidden(int id) { if(id < hidden.size()) { hidden.erase(hidden.begin() + id); compiled = false; } }
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