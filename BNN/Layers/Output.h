#pragma once
#include "Layer.h"
namespace BNN {
	//Output layer, allows reshaping, provides error calculation (output gradient)
	class Output : public Layer {
	public:
		Output(shp3 d, Layer* prev = nullptr, Afun af = Afun::t_lin, Efun ef = Efun::t_mse) : Layer(d, prev), ef(ef), af(af) {}
		Output(Layer* prev, Afun af = Afun::t_lin, Efun ef = Efun::t_mse) : Layer(prev->dim_y(), prev), ef(ef), af(af) {}
		const Tensor& output() const { return y(); }
		void derivative() override {
			x() = y().reshape(x().dimensions());
		}
		float error(const Tensor& y0) {
			float cost = 0;
			cost = fsca(y().binaryExpr(y0, ef.fx()).mean()).coeff();
			x() = y().binaryExpr(y0, ef.dx()).reshape(x().dimensions());
			return cost;
		}
		bool compile(Layer* pnode, Layer* nnode) override {
			set_prev(pnode);
			set_next(nullptr);
			return in_eq_out();
		}
		void print()const override {
			println("Output\t|", "\tIn:", dim_x(0), dim_x(1), dim_x(2), "\tOut:", dim_y(0), dim_y(1), dim_y(2));
		}
		Output* clone() const override { return new Output(*this); }
	private:
		Tensor compute(const Tensor& x) const override {
			return x.reshape(dim_y()).unaryExpr(af.fx());
		}
		const Tensor& predict() override { return y() = x().reshape(dim_y()).unaryExpr(af.fx()); }
		Efun ef;
		Afun af;
	};
}