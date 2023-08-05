#pragma once
#include "GD_nodes.h"
namespace BNN {
#define COMMON_OPTIM_FUNCS(TYPE) \
private:\
vector<TYPE> nodes;\
	void build(Layer* node) {\
		if (!node||!node->prev) return; \
		Layer* curr = node->prev;\
		while (curr->prev) {\
			nodes.push_back(curr); \
			curr = curr->prev;\
		}\
	};\
public:\
void compile(Layer* last) override {\
	nodes.clear();\
	build(last);\
};\
void get_grad() override { \
	for (auto& n : nodes) {\
		n.get_grad(inv_n);\
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
		virtual void compile(Layer* last) = 0;
		virtual void get_grad() = 0;
		virtual void update_grad() = 0;
		virtual void reset_grad() = 0;
		virtual void reset_cache() = 0;
		virtual void reset_all() = 0;
		virtual void print() = 0;
		float alpha = 0.001f, inv_n = 1.f;
	};
	class SGD : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::SGD_node)
			SGD(float alpha, Layer* last = nullptr) : Optimizer(alpha) { build(last); }
		void update_grad() override {
			for (auto& n : nodes) {
				n.update_grad(alpha);
			}
		};
		void print() override {
			println("SGD", "\tRate:", alpha, "\tNodes:", nodes.size());
		}
	};
	class AGD : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::AGD_node)
			AGD(float alpha, Layer* last = nullptr) : Optimizer(alpha) { build(last); }
			AGD(float alpha,float mu, Layer* last = nullptr) : Optimizer(alpha),mu(mu) { build(last); }
		void update_grad() override {
			for (auto& n : nodes) {
				n.update_grad(alpha, mu);
			}
		};
		void print() override {
			println("AGD", "\tRate:", alpha, "\tMu:", mu, "\tNodes:", nodes.size());
		}
		float mu = 0.9f;
	};
	class NAG : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::NAG_node)
			NAG(float alpha, Layer* last = nullptr) : Optimizer(alpha) { build(last); }
			NAG(float alpha, float mu, Layer* last = nullptr) : Optimizer(alpha), mu(mu) { build(last); }
		void update_grad() override {
			for (auto& n : nodes) {
				n.update_grad(alpha, mu);
			}
		};
		void print() override {
			println("NAG", "\tRate:", alpha, "\tMu:", mu, "\tNodes:", nodes.size());
		}
		float mu = 0.9f;
	};
	class RMSprop : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::RMS_node)
			RMSprop(float alpha, Layer* last = nullptr) : Optimizer(alpha) { build(last); }
		RMSprop(float alpha, float b, float eps, Layer* last = nullptr) : Optimizer(alpha), beta(b), eps(eps) { build(last); }
		void update_grad() override {
			for (auto& n : nodes) {
				n.update_grad(alpha, beta, eps);
			}
		};
		void print() override {
			println("RMSprop", "\tRate:", alpha, "\tBeta:", beta, "\tNodes:", nodes.size());
		}
		float beta = 0.9f, eps = 1e-6f;
	};

	class Adam : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::ADAM_node)
			Adam(float alpha, float b1, float b2, float eps, Layer* last = nullptr) : Optimizer(alpha), beta1(b1), beta2(b2), eps(eps) { build(last); }
		Adam(float alpha, Layer* last = nullptr) : Optimizer(alpha) { build(last); }
		void update_grad() override {
			for (auto& n : nodes) {
				n.update_grad(alpha, beta1, beta2, eps);
			}
		};
		void print() override {
			println("Adam", "\tRate:", alpha, "\tBeta:", beta1, beta2, "\tNodes:", nodes.size());
		}
		float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-6f;
	};
}