#pragma once
#include "Layer.h"
namespace BNN {
	//Can be used for either pooling or downsampling
	class AvgPool : public Layer {
	public:
		AvgPool(shp3 din, shp2 ks, shp2 st, shp2 pa, Layer* prev = nullptr) :
			Layer({ din[0],c_dim(din[1],ks[0],st[0],pa[0]),c_dim(din[2],ks[1],st[1],pa[1]) }, prev),
			din(din), ks(ks), st(st), pa(pa) {
		}
		AvgPool(shp2 ks, shp2 st, shp2 pa, Layer* prev) :
			Layer({ prev->dim_y(0),c_dim(prev->dim_y(1),ks[0],st[0],pa[0]),c_dim(prev->dim_y(2),ks[1],st[1],pa[1]) }, prev),
			din(prev->dim_y()), ks(ks), st(st), pa(pa) {
		}
		void derivative() override {
			auto dy = y().inflate(dim1<3>{ 1, st[0], st[1] });
			idx p1 = c_pad(din[1], ks[0], st[0], dim_y(1));
			idx p2 = c_pad(din[2], ks[1], st[1], dim_y(2));
			x() = pool_avg(dy, ks, 1, { p1, p2 }).reshape(dim_x());
		}
		idx sz_in() const override { return product(din); }
		dim1<3> dim_in() const override { return din; }
		void print()const override {
			println("AvgPl\t|", "\tIn:", din[0], din[1], din[2],
				"\tOut:", dim_y(0), dim_y(1), dim_y(2), "\tKernel:", dim_w(1), dim_w(2), "\tStride:", st[0], st[1], "\tPad:", pa[0], pa[1]);
		}
		AvgPool* clone() const override { return new AvgPool(*this); }
	private:
		Tensor compute(const Tensor& x) const override {
			return pool_avg(x.reshape(din), ks, st, pa);
		}
		const Tensor& predict() override {
			pool_avg_r(y(), x().reshape(din),ks,st, pa);
			return next->predict();
		}
		dim1<3> din;
		shp2 ks, st, pa;
	};
}