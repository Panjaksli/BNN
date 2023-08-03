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
				return false;
			}
			else {
				input->compile(nullptr, hidden[0]);
				for (idx i = 0; i < hidden.size(); i++) {
					if (!hidden[i]->compile(i == 0 ? input : hidden[i - 1], i < hidden.size() - 1 ? hidden[i + 1] : output)) {
						println("Error:  Dimension mismatch in node: ", i, "!");
						return false;
					}
				}
			}
			if (!output->compile(hidden.back(), nullptr)) {
				println("Error:  Dimension mismatch in output node !");
				return false;
			}
			optimizer->compile(output);
			println("Message:  Network compiled succesfully:");
			print();
			return true;
		}
		void print() {
			println("------------------------------------------------------------------------------------------------");
			std::cout << "Network\t|\t"; println(name);
			if(input)input->print();
			for (const auto& h : hidden) h->print();
			if(output)output->print();
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
	};
}