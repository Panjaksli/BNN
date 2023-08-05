#pragma once
#include "Layer.h"
namespace BNN {
	//Dropout layer, allows reshaping
	class Dropout : public Layer {
	public:
		Dropout(float rate, Layer* prev) : Layer(prev->dim_y(), prev), w(prev->dim_y()), rate(rate) { _init(); }
		Dropout(float rate, shp3 din, Layer* prev = nullptr) : Layer(din , prev), w(din), rate(rate) { _init(); }
		void init() override { _init(); }
		void derivative() override {
			x() = (y() * w).reshape(dim_x());
		}
		idx sz_in() const override { return w.size(); }
		dim1<3> dim_in() const override { return w.dimensions(); }
		void print()const override {
			println("Drop\t|", "\tIn:", dim_x(0), dim_x(1), dim_x(2),
				"\tOut:", dim_y(0), dim_y(1), dim_y(2));
		}
	private:
		Tensor compute(const Tensor& x) const override {
			return x.reshape(dim_in()) * w;
		}
		const Tensor& predict() override {
			y() = x().reshape(dim_in()) * w;
			return next->predict();
		}
		void update() override { _init(); }
		void _init() {
			float weight = (1.f / (1.f - rate));
			w = w.random().unaryExpr([weight, this](float x) {return (x > rate) * weight; });
		}
		Tensor w;
		float rate = 0.1f;
	};
}