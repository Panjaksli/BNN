#pragma once
#include "Layer.h"
namespace BNN {
	//Shape layer, allows reshaping
	class Shape : public Layer {
	public:
		Shape(Layer* prev) : Layer(prev->odims(), prev) {}
		Shape(shp3 din, Layer* prev = nullptr) : Layer(din, prev) {}
		void init() override { }
		void derivative(bool ptrain) override {
			if(ptrain) x() = y().reshape(pdims());
		}
		void print()const override {
			println("Shape\t|", "\tDim:", odim(0), odim(1), odim(2));
		}
		void save(std::ostream& out)const override {
			out << "Hidden Shape" SPC odim(0) SPC odim(1) SPC odim(2) << "\n";
			next->save(out);
		}
		static auto load(std::istream& in) {
			shp3 d;
			in >> d[0] >> d[1] >> d[2];
			return new Shape(d);
		}
		Shape* clone() const override { return new Shape(*this); }
		LType type() const override { return t_Shape; }
	private:
		Tensor compute(const Tensor& x) const override {
			return next->compute(x.reshape(idims()));
		}
		Tensor compute_ds(const Tensor& x) const override {
			dim1<3> dims{ x.dimension(0) * odim(0) / pdim(0), x.dimension(1) * odim(1) / pdim(1), x.dimension(2) * odim(2) / pdim(2) };
			return next->compute(x.reshape(dims));
		}
		const Tensor& predict() override {
			y() = x().reshape(idims());
			return next->predict();
		}
		const Tensor& predict(const Tensor& x) override {
			return next->predict(y() = x.reshape(idims()));
		}
	};
}