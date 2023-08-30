#pragma once
#include "Layer.h"
namespace BNN {
	//Output layer, allows reshaping, provides error calculation (output gradient)
	class Output : public Layer {
	public:
		Output() {}
		Output(shp3 d, Layer* prev = nullptr, Efun ef = Efun::t_mse) : Layer(d, prev), ef(ef) {}
		Output(Layer* prev, Efun ef = Efun::t_mse) : Layer(prev->odims(), prev), ef(ef) {}
		const Tensor& output() const { return y(); }
		void derivative(bool ptrain) override {
			if(ptrain) x() = y().reshape(pdims());
		}
		float error(const Tensor& y0) override {
			float cost = 0;
			cost = fsca(y().binaryExpr(y0, &Efun::mae::fx).mean()).coeff();
			x() = y().binaryExpr(y0, ef.dx()).reshape(pdims());
			return cost;
		}
		void print()const override {
			println("Output\t|", "\tDim:", odim(0), odim(1), odim(2), ef.name());
		}
		void save(std::ostream& out)const override {
			out << "Hidden Output" SPC odim(0) SPC odim(1) SPC odim(2) SPC ef.type << "\n";
		}
		static auto load(std::istream& in) {
			shp3 d; idx ef;
			in >> d[0] >> d[1] >> d[2] >> ef;
			return new Output(d, nullptr, (Efun::Type)ef);
		}
		Output* clone() const override { return new Output(*this); }
		LType type() const override { return t_Output; }
	private:
		Tensor compute(const Tensor& x) const override {
			return x.reshape(odims());
		}
		Tensor compute_ds(const Tensor& x) const override {
			return x;
		}
		const Tensor& predict(const Tensor& x) override {
			return y() = x.reshape(odims());
		}
		const Tensor& predict() override {
			return y() = x().reshape(odims());
		}
		Efun ef;
	};
}