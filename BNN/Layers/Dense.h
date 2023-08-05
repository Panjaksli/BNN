#pragma once
#include "Layer.h"
namespace BNN {
	class Dense : public Layer {
	public:
		Dense(shp2 d, Afun af = Afun::t_lrelu) : Dense(d, nullptr, af) {}
		Dense(shp2 d, Layer* prev, Afun af = Afun::t_lrelu) :
			Layer({ 1,d[0],1 }, prev), dz(dim_y()), b(dim_y()),
			w(1, d[1], prev ? sz_x() : d[0]), af(af) {
			_init();
		}
		void init() override { _init(); }
		void derivative() override {
			dz = y() * dz;
			mul_r(x().reshape(dim1<3>{ 1, dim_w(2), 1 }), w, dz, {0,0});
		}
		void gradient(Tensor& dw, Tensor& db, float inv_n = 1.f) override {
			dz = y() * dz;
			db += dz * inv_n;
			dw += mul(dz, x().reshape(dim1<3>{ 1, 1, dim_w(2) }), { 1,0 })* inv_n;
			mul_r(x().reshape(dim1<3>{ 1, dim_w(2), 1 }), w, dz, { 0,0 });
		}
		Tensor* get_w() override { return &w; }
		Tensor* get_b() override { return &b; }
		dim1<3> dim_w() const override { return w.dimensions(); }
		dim1<3> dim_b() const override { return b.dimensions(); }
		idx dim_w(idx i) const override { return w.dimension(i); }
		idx dim_b(idx i) const override { return b.dimension(i); }
		idx sz_in() const override { return w.dimension(2); }
		dim1<3> dim_in() const override { return dim1<3>{1,w.dimension(2),1}; }
		void print()const override {
			println("Dense\t|", "\tIn:", 1, dim_w(2), 1, "\tOut:", dim_y(0), dim_y(1), dim_y(2));
		}
	private:
		void _init() {
			b = b.random() * 0.5f - 0.25f;
			w = w.random() * 0.5f - 0.25f;
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute(fma(w, x.reshape(dim1<3>{ 1, dim_w(2), 1 }), b).unaryExpr(af.fx()));
		}
		const Tensor& predict() override {
			fma_r(y(), w, x().reshape(dim1<3>{ 1, dim_w(2), 1 }), b);
			dz = y().unaryExpr(af.dx());
			y() = y().unaryExpr(af.fx());
			return next->predict();
		}
		Tensor dz, b;
		Tensor w;
		Afun af;
	};
}