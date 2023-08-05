#pragma once
#include "Layer.h"
namespace BNN {
	//Allows reshaping of the input
	class Input : public Layer {
	public:
		Input(shp3 d) : Layer(d) {}
		void input(const Tensor& x) {
			y() = x.reshape(dim_y());
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute(x.reshape(dim_y()));
		}
		Tenarr compute_batch(const Tenarr& x) const {
			dim1<3> d = last()->dim_y();
			Tenarr y(x.dimension(0), d[0], d[1], d[2]);
			for (int i = 0; i < x.dimension(0); i++) {
				y.chip(i, 0) = compute(x.chip(i, 0));
			}
			return y;
		}
		const Tensor& predict(const Tensor& x) {
			input(x);
			return next->predict();
		}
		Tenarr predict_batch(const Tenarr& x) {
			dim1<3> d = last()->dim_y();
			Tenarr y(x.dimension(0), d[0], d[1], d[2]);
			for (int i = 0; i < x.dimension(0); i++) {
				y.chip(i, 0) = predict(x.chip(i, 0));
			}
			return y;
		}
		bool compile(Layer* pnode, Layer* nnode) override {
			set_prev(nullptr);
			set_next(nnode);
			return true;
		}
		void print()const override {
			println("Input\t|", "\tIn:", dim_y(0), dim_y(1), dim_y(2), "\tOut:", dim_y(0), dim_y(1), dim_y(2));
		}
		Input* clone() const override { return new Input(*this); }
	private:
		const Tensor& predict() override { return next->predict(); }
	};
}