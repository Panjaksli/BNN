#pragma once
#include "Layer.h"
namespace BNN {
	//Output layer, allows reshaping, provides error calculation (output gradient)
	class Output : public Layer {
	public:
		Output() {}
		Output(shp3 d, Layer* prev = nullptr, Afun af = Afun::t_lin, Efun ef = Efun::t_mse) : Layer(d, prev), ef(ef), af(af) {}
		Output(Layer* prev, Afun af = Afun::t_lin, Efun ef = Efun::t_mse) : Layer(prev->odims(), prev), ef(ef), af(af) {}
		const Tensor& output() const { return y(); }
		void derivative() override {
			x() = y().reshape(pdims());
		}
		float error(const Tensor& y0) {
			float cost = 0;
			cost = fsca(y().binaryExpr(y0, ef.fx()).mean()).coeff();
			x() = y().binaryExpr(y0, ef.dx()).reshape(pdims());
			return cost;
		}
		bool compile(Layer* pnode, Layer* nnode) override {
			set_prev(pnode);
			set_next(nullptr);
			return in_eq_out();
		}
		void print()const override {
			println("Output\t|", "\tIn:", pdim(0), pdim(1), pdim(2), "\tOut:", odim(0), odim(1), odim(2));
		}
		void save(std::ostream& out)const override {
			out << "Output" SPC odim(0) SPC odim(1) SPC odim(2) SPC af.type SPC ef.type << "\n";
		}
		static auto load(std::istream& in) {
			shp3 d; int af; int ef;
			in >> d[0] >> d[1] >> d[2] >> af >> ef;
			return new Output(d, nullptr, (Afun::Type)af, (Efun::Type)ef);
		}
		Output* clone() const override { return new Output(*this); }
	private:
		Tensor compute(const Tensor& x) const override {
			return x.reshape(odims()).unaryExpr(af.fx());
		}
		const Tensor& predict() override { return y() = x().reshape(odims()).unaryExpr(af.fx()); }
		Efun ef;
		Afun af;
	};
}