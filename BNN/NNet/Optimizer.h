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
	for (idx i = 0 ; i < nodes.size(); i++) {\
		nodes[i].get_grad(i < nodes.size() - 1);\
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
}\
idx size() override { return nodes.size(); }\
Tensor* get_vw(idx i) override { return nodes[i].get_vw(); } \
Tensor* get_vb(idx i) override { return nodes[i].get_vb(); }\
Tensor* get_dw(idx i) override { return nodes[i].get_dw(); } \
Tensor* get_db(idx i) override { return nodes[i].get_db(); }\
Tensor* get_mw(idx i) override { return nodes[i].get_mw(); }\
Tensor* get_mb(idx i) override { return nodes[i].get_mb(); }


	class Optimizer {
	public:
		Optimizer() {}
		Optimizer(float alpha, Regular reg) : alpha(alpha), reg(reg) {}
		virtual ~Optimizer() {}
		virtual void compile(Layer* first) {}
		virtual void get_grad() {}
		virtual void update_grad() {}
		virtual void reset_grad() {}
		virtual void reset_cache() {}
		virtual void reset_all() {}
		virtual void print() const { println("None"); }
		virtual void save(std::ostream& out) const { out << "Optimizer" SPC "None" SPC reg << "\n"; }
		virtual idx size() { return 0; }
		virtual Tensor* get_vw(idx i) { return nullptr; }
		virtual Tensor* get_vb(idx i) { return nullptr; }
		virtual Tensor* get_mw(idx i) { return nullptr; }
		virtual Tensor* get_mb(idx i) { return nullptr; }
		virtual Tensor* get_dw(idx i) { return nullptr; }
		virtual Tensor* get_db(idx i) { return nullptr; }
		virtual Optimizer* clone() const { return new Optimizer(); }
		float alpha = 0.001f, inv_n = 1.f;
		Regular reg = Regular();
	};
	class SGD : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::SGD_node)
			SGD() {}
		SGD(float alpha, Regular reg = Regular(), Layer* first = nullptr) : Optimizer(alpha, reg) { build(first); }
		void update_grad() override {
			for(auto& n : nodes) {
				n.update_grad(alpha, inv_n, reg);
			}
		};
		void print() const override {
			println("SGD", "\tRate:", alpha, "\tNodes:", nodes.size());
		}
		void save(std::ostream& out) const override {

			out << "Optimizer" SPC "SGD" SPC alpha SPC reg << "\n";
		}
		static auto load(std::istream& in) {
			float a, l1, l2;
			int r;
			in >> a >> r >> l1 >> l2;
			return new SGD(a, Regular(RegulTag(r), l1, l2));
		}
		virtual SGD* clone() const override { return new SGD(*this); };
	};
	class AGD : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::AGD_node)
			AGD() {}
		AGD(float alpha, Regular reg = Regular(), Layer* first = nullptr) : Optimizer(alpha, reg) { build(first); }
		AGD(float alpha, float mu, Regular reg = Regular(), Layer* first = nullptr) : Optimizer(alpha, reg), mu(mu) { build(first); }
		void update_grad() override {
			for(auto& n : nodes) {
				n.update_grad(alpha, mu, inv_n, reg);
			}
		};
		void print() const override {
			println("AGD", "\tRate:", alpha, "\tMu:", mu, "\tNodes:", nodes.size());
		}
		void save(std::ostream& out) const override {
			out << "Optimizer" SPC "AGD" SPC alpha SPC mu SPC reg << "\n";
		}
		static auto load(std::istream& in) {
			float a, m, l1, l2;
			int r;
			in >> a >> m >> r >> l1 >> l2;
			return new AGD(a, m, Regular(RegulTag(r), l1, l2));
		}
		virtual AGD* clone() const override { return new AGD(*this); };
		float mu = 0.9f;
	};
	class NAG : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::NAG_node)
			NAG() {}
		NAG(float alpha, Regular reg = Regular(), Layer* first = nullptr) : Optimizer(alpha, reg) { build(first); }
		NAG(float alpha, float mu, Regular reg = Regular(), Layer* first = nullptr) : Optimizer(alpha, reg), mu(mu) { build(first); }
		void update_grad() override {
			for(auto& n : nodes) {
				n.update_grad(alpha, mu, inv_n, reg);
			}
		};
		void print() const override {
			println("NAG", "\tRate:", alpha, "\tMu:", mu, "\tNodes:", nodes.size());
		}
		void save(std::ostream& out) const override {
			out << "Optimizer" SPC"NAG" SPC alpha SPC mu  SPC reg << "\n";
		}
		static auto load(std::istream& in) {
			float a, m, l1, l2;
			int r;
			in >> a >> m >> r >> l1 >> l2;
			return new NAG(a, m, Regular(RegulTag(r), l1, l2));
		}
		virtual NAG* clone() const override { return new NAG(*this); };
		float mu = 0.9f;
	};
	class RMSprop : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::RMS_node)
			RMSprop() {}
		RMSprop(float alpha, Regular reg = Regular(), Layer* first = nullptr) : Optimizer(alpha, reg) { build(first); }
		RMSprop(float alpha, float b, float eps, Regular reg = Regular(), Layer* first = nullptr) : Optimizer(alpha, reg), beta(b), eps(eps) { build(first); }
		void update_grad() override {
			for(auto& n : nodes) {
				n.update_grad(alpha, beta, eps, inv_n, reg);
			}
		};
		void print() const override {
			println("RMSprop", "\tRate:", alpha, "\tBeta:", beta, "\tEps", eps, "\tNodes:", nodes.size());
		}
		void save(std::ostream& out) const override {
			out << "Optimizer" SPC "RMSprop" SPC alpha SPC beta SPC eps  SPC reg << "\n";
		}
		static auto load(std::istream& in) {
			float a, b, e, l1, l2;
			int r;
			in >> a >> b >> e >> r >> l1 >> l2;
			return new RMSprop(a, b, e, Regular(RegulTag(r), l1, l2));
		}
		virtual RMSprop* clone() const override { return new RMSprop(*this); };
		float beta = 0.9f, eps = 1e-8f;
	};

	class Adam : public Optimizer {
		COMMON_OPTIM_FUNCS(GD::ADAM_node)
			Adam() {}
		Adam(float alpha, float b1, float b2, float eps, Regular reg = Regular(), Layer* first = nullptr) : Optimizer(alpha, reg), beta1(b1), beta2(b2), eps(eps) { build(first); }
		Adam(float alpha, Regular reg = Regular(), Layer* first = nullptr) : Optimizer(alpha, reg) { build(first); }
		void update_grad() override {
			for(auto& n : nodes) {
				n.update_grad(alpha, beta1, beta2, eps, inv_n, reg);
			}
		};
		void save(std::ostream& out) const override {
			out << "Optimizer" SPC "Adam" SPC alpha SPC beta1 SPC beta2 SPC eps SPC reg << "\n";
		}
		static auto load(std::istream& in) {
			float a, b1, b2, e, l1, l2;
			int r;
			in >> a >> b1 >> b2 >> e >> r >> l1 >> l2;
			return new Adam(a, b1, b2, e, Regular(RegulTag(r), l1, l2));
		}
		void print() const override {
			println("Adam", "\tRate:", alpha, "\tBeta:", beta1, beta2, "\tEps", eps, "\tNodes:", nodes.size());
		}
		virtual Adam* clone() const override { return new Adam(*this); };
		float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
	};
	inline Optimizer* Optimizer_load(std::istream& in) {
		std::string token;
		in >> token;
		if(token == "SGD") return SGD::load(in);
		else if(token == "AGD") return AGD::load(in);
		else if(token == "NAG") return NAG::load(in);
		else if(token == "RMSprop") return RMSprop::load(in);
		else if(token == "Adam") return Adam::load(in);
		else return new Optimizer();
	}
}

#undef COMMON_OPTIM_FUNCS