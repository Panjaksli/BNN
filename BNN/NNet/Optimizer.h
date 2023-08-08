#pragma once
#include "GD_nodes.h"
namespace BNN {
#define COMMON_OPTIM_FUNCS(TYPE) \
private:\
vector<TYPE> nodes;\
void build(Layer* node) {\
	if(!node) return;\
	nodes.clear();\
	while(!node->trainable() && node->next) node = node->next;\
	while(node->next) {\
		nodes.push_back(node);\
		node = node->next;\
	}\
	std::reverse(nodes.begin(), nodes.end());\
}\
public:\
void compile(Layer* first) override {\
	build(first);\
};\
void get_grad() override { \
	for (int i = 0 ; i < nodes.size(); i++) {\
		nodes[i].get_grad(i < nodes.size() - 1, inv_n);\
	}\
};\
void reset_grad() override {\
	for (auto& n : nodes) {\
		n.reset_grad();\
	}\
}\
void reset_cache() override {\
	for (auto& n : nodes) {\
		n.reset_cache();\
	}\
}\
void reset_all() override {\
	for (auto& n : nodes) {\
		n.reset_grad();\
		n.reset_cache();\
	}\
}
	class Optimizer {
	public:
		Optimizer() {}
		Optimizer(float alpha) : alpha(alpha) {}
		virtual ~Optimizer() {}
		virtual void compile(Layer* first) {}
		virtual void get_grad() {}
		virtual void update_grad() {}
		virtual void reset_grad() {}
		virtual void reset_cache() {}
		virtual void reset_all() {}
		virtual void print() { println("None"); }
		virtual void save(std::ostream& out) { out << "Optimizer" SPC "None" << "\n"; }
		virtual Optimizer* clone() const { return new Optimizer(); }
		float alpha = 0.001f, inv_n = 1.f;
	};
	class SGD : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::SGD_node)
			SGD() {}
		SGD(float alpha, Layer* first = nullptr) : Optimizer(alpha) { build(first); }
		void update_grad() override {
			for(auto& n : nodes) {
				n.update_grad(alpha);
			}
		};
		void print() override {
			println("SGD", "\tRate:", alpha, "\tNodes:", nodes.size());
		}
		void save(std::ostream& out) override {

			out << "Optimizer" SPC "SGD" SPC alpha << "\n";
		}
		static auto load(std::istream& in) {
			float a;
			in >> a;
			return new SGD(a);
		}
		virtual SGD* clone() const override { return new SGD(*this); };
	};
	class AGD : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::AGD_node)
			AGD() {}
		AGD(float alpha, Layer* first = nullptr) : Optimizer(alpha) { build(first); }
		AGD(float alpha, float mu, Layer* first = nullptr) : Optimizer(alpha), mu(mu) { build(first); }
		void update_grad() override {
			for(auto& n : nodes) {
				n.update_grad(alpha, mu);
			}
		};
		void print() override {
			println("AGD", "\tRate:", alpha, "\tMu:", mu, "\tNodes:", nodes.size());
		}
		void save(std::ostream& out) override {
			out << "Optimizer" SPC"AGD" SPC alpha SPC mu << "\n";
		}
		static auto load(std::istream& in) {
			float a, m;
			in >> a >> m;
			return new AGD(a, m);
		}
		virtual AGD* clone() const override { return new AGD(*this); };
		float mu = 0.9f;
	};
	class NAG : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::NAG_node)
			NAG() {}
		NAG(float alpha, Layer* first = nullptr) : Optimizer(alpha) { build(first); }
		NAG(float alpha, float mu, Layer* first = nullptr) : Optimizer(alpha), mu(mu) { build(first); }
		void update_grad() override {
			for(auto& n : nodes) {
				n.update_grad(alpha, mu);
			}
		};
		void print() override {
			println("NAG", "\tRate:", alpha, "\tMu:", mu, "\tNodes:", nodes.size());
		}
		void save(std::ostream& out) override {
			out << "Optimizer" SPC"NAG" SPC alpha SPC mu << "\n";
		}
		static auto load(std::istream& in) {
			float a, m;
			in >> a >> m;
			return new NAG(a, m);
		}
		virtual NAG* clone() const override { return new NAG(*this); };
		float mu = 0.9f;
	};
	class RMSprop : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::RMS_node)
			RMSprop() {}
		RMSprop(float alpha, Layer* first = nullptr) : Optimizer(alpha) { build(first); }
		RMSprop(float alpha, float b, float eps, Layer* first = nullptr) : Optimizer(alpha), beta(b), eps(eps) { build(first); }
		void update_grad() override {
			for(auto& n : nodes) {
				n.update_grad(alpha, beta, eps);
			}
		};
		void print() override {
			println("RMSprop", "\tRate:", alpha, "\tBeta:", beta, "\tEps", eps, "\tNodes:", nodes.size());
		}
		void save(std::ostream& out) override {
			out << "Optimizer" SPC "RMSprop" SPC alpha SPC beta SPC eps << "\n";
		}
		static auto load(std::istream& in) {
			float a, b, e;
			in >> a >> b >> e;
			return new RMSprop(a, b, e);
		}
		virtual RMSprop* clone() const override { return new RMSprop(*this); };
		float beta = 0.9f, eps = 1e-6f;
	};

	class Adam : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::ADAM_node)
			Adam() {}
		Adam(float alpha, float b1, float b2, float eps, Layer* first = nullptr) : Optimizer(alpha), beta1(b1), beta2(b2), eps(eps) { build(first); }
		Adam(float alpha, Layer* first = nullptr) : Optimizer(alpha) { build(first); }
		void update_grad() override {
			for(auto& n : nodes) {
				n.update_grad(alpha, beta1, beta2, eps);
			}
		};
		void save(std::ostream& out) override {
			out << "Optimizer" SPC "Adam" SPC alpha SPC beta1 SPC beta2 SPC eps << "\n";
		}
		static auto load(std::istream& in) {
			float a, b1, b2, e;
			in >> a >> b1 >> b2 >> e;
			return new Adam(a, b1, b2, e);
		}
		void print() override {
			println("Adam", "\tRate:", alpha, "\tBeta:", beta1, beta2, "\tEps", eps, "\tNodes:", nodes.size());
		}
		virtual Adam* clone() const override { return new Adam(*this); };
		float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-6f;
	};
	inline Optimizer* Optimizer_load(std::istream& in) {
		std::string token;
		in >> token;
		if(token == "SGD") return SGD::load(in);
		else if(token == "AGD") return AGD::load(in);
		else if(token == "NAG") return NAG::load(in);
		else if(token == "Rmsprop") return RMSprop::load(in);
		else if(token == "Adam") return Adam::load(in);
		else return new Optimizer();
	}
}

#undef COMMON_OPTIM_FUNCS