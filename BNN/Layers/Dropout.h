#pragma once
#include "Layer.h"
namespace BNN {
	//Dropout layer, allows reshaping
	class Dropout : public Layer {
	public:
		Dropout(float rate, Layer* prev) : Layer(prev->odims(), prev), dz(pdims()), rate(rate) { _init(); }
		Dropout(float rate, shp3 din, Layer* prev = nullptr) : Layer(din, prev), dz(din), rate(rate) { _init(); }
		void init() override { _init(); }
		void derivative(bool ptrain) override {
			if(ptrain) x() = (y() * dz).reshape(pdims());
		}
		dim1<3> idims() const override { return dz.dimensions(); }
		void print()const override {
			println("Dropout\t|", "\tIn:", pdim(0), pdim(1), pdim(2),
				"\tOut:", odim(0), odim(1), odim(2), "\tRate:", rate);
		}
		void save(std::ostream& out)const override {
			out << "Hidden Dropout" SPC odim(0) SPC odim(1) SPC odim(2) SPC rate << "\n";
			next->save(out);
		}
		static auto load(std::istream& in) {
			shp3 d; float rate;
			in >> d[0] >> d[1] >> d[2] >> rate;
			return new Dropout(rate, d);
		}
		Dropout* clone() const override { return new Dropout(*this); }
		LType type() const override { return t_Dropout; }
	private:
		Tensor compute(const Tensor& x) const override {
			//We should skip the dropout during normal computation !
			return next->compute(x);
		}
		Tensor comp_dyn(const Tensor& x) const override {
			return next->compute(x);
		}
		const Tensor& predict() override {
			y() = x().reshape(idims()) * dz;
			return next->predict();
		}
		const Tensor& predict(const Tensor& x) override {
			return next->predict(y() = x.reshape(idims()) * dz);
		}
		void update() override { _init(); }
		void _init() {
			float weight = (1.f / (1.f - rate));
			dz = dz.random().unaryExpr([weight, this](float x) {return (x > rate) * weight; });
		}
		Tensor dz;
		float rate = 0.1f;
	};
}