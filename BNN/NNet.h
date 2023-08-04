#pragma once
#include "Optimizer.h"
#include "Layer.h"
#include "Afun.h"
namespace BNN {
	class NNet {
	public:
		~NNet() {
			if (input)delete input;
			if (output)delete output;
			if (optimizer)delete optimizer;
			for (auto& h : hidden) delete h;
		}
		NNet() :input(nullptr), output(nullptr), optimizer(nullptr) {}
		NNet(Input* in, Output* out, Optimizer* opt) : input(in), output(out), optimizer(opt) {

		}
		NNet(Input* in, const vector<Layer*>& hid, Output* out, Optimizer* opt) : input(in), output(out), hidden(hid), optimizer(opt) {
			Compile();
		}
		bool Compile() {
			if (!input || !output || hidden.size() == 0) {
				println("Error:  Incomplete model ! (Node is missing...)");
				return compiled = false;
			}
			else {
				input->compile(nullptr, hidden[0]);
				for (idx i = 0; i < hidden.size(); i++) {
					if (!hidden[i]->compile(i == 0 ? input : hidden[i - 1], i < hidden.size() - 1 ? hidden[i + 1] : output)) {
						println("Error:  Dimension mismatch in node: ", i, "!");
						return compiled = false;
					}
				}
			}
			if (!output->compile(hidden.back(), nullptr)) {
				println("Error:  Dimension mismatch in output node !");
				return compiled = false;
			}
			optimizer->compile(output);
			println("Message:  Network compiled succesfully:");
			print();
			return compiled = true;
		}
		bool train(const Tenarr& x0, const Tenarr& y0, float rate = 0.001f, idx epochs = 1000, int nlog = 100) {
			if (!valid()) return false;
			optimizer->alpha = rate;
			optimizer->inv_n = 1.f / x0.dimension(0);
			optimizer->reset_all();
			double t = timer();
			double dt = timer();
			float inv_ep = 1.f / epochs;
			int log_step = epochs / nlog;
			log_step = max(1, log_step);
			for (int i = 1; i <= epochs; i++) {
				float cost = 0;
				for (int j = 0; j < x0.dimension(0); j++) {
					input->predict(x0.chip(j, 0));
					cost += output->error(y0.chip(j, 0)) * inv_ep;
					optimizer->get_grad();
				}
				optimizer->update_grad();
				if (i % log_step == 0) {
					printr('\0', "Epoch:", i, "Cost:", cost, "Time:", timer(t), "Step:", timer(dt) / log_step); 
					dt = timer();
				}
			}
			return true;
		}
		bool valid() {
			return input && output && optimizer && hidden.size() > 0 && compiled;
		}
		void print() {
			println("------------------------------------------------------------------------------------------------");
			std::cout << "Network\t|\t"; println(name);
			if (input)input->print();
			for (const auto& h : hidden) h->print();
			if (output)output->print();
			std::cout << ("Optimiz\t|\t"); if (optimizer) { optimizer->print(); }
			println("------------------------------------------------------------------------------------------------");
		}
		void Set_input(Input* in) { input = in; }
		void Set_output(Output* out) { output = out; }
		void Add_hidden(Layer* hidl) { hidden.push_back(hidl); }
	public:
		Input* input;
		Output* output;
		vector<Layer*> hidden;
		Optimizer* optimizer;
		std::string name = "net";
		bool compiled = false;
	};
}