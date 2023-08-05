#pragma once
#pragma once
#include "Layer.h"
namespace BNN {
	//Can be used for either unpooling or upsampling
	class AvgUpool : public Layer {
	public:
		AvgUpool(shp3 din, shp2 ks, shp2 st, shp2 pa, Layer* prev = nullptr) :
			Layer({ din[0],t_dim(din[1],ks[0],st[0],pa[0]),t_dim(din[2],ks[1],st[1],pa[1]) }, prev),
			din(din), ks(ks), st(st), pa(pa) {
		}
		AvgUpool(shp2 ks, shp2 st, shp2 pa, Layer* prev) :
			Layer({ prev->dim_y(0),t_dim(prev->dim_y(1),ks[0],st[0],pa[0]),t_dim(prev->dim_y(2),ks[1],st[1],pa[1]) }, prev),
			din(prev->dim_y()), ks(ks), st(st), pa(pa) {
		}
		void derivative() override {
			idx p1 = t_pad(din[1], ks[0], st[0], dim_y(1));
			idx p2 = t_pad(din[2], ks[1], st[1], dim_y(2));
			x() = pool_avg(y(), ks, st, { p1,p2 }).reshape(dim_x());
		}
		idx sz_in() const override { return din[0] * din[1] * din[2]; }
		dim1<3> dim_in() const override { return din; }
		void print()const override {
			println("AvgUpl\t|", "\tIn:", din[0], din[1], din[2],
				"\tOut:", dim_y(0), dim_y(1), dim_y(2), "\tKernel:", dim_w(1), dim_w(2), "\tStride:", st[0], st[1], "\tPad:", pa[0], pa[1]);
		}
	private:
		Tensor compute(const Tensor& x) const override {
			return pool_avg(x.reshape(din).inflate(dim1<3>{ 1, st[0], st[1] }), ks, 1, ks - pa - 1);
		}
		const Tensor& predict() override {
			auto ix = x().reshape(din).inflate(dim1<3>{ 1, st[0], st[1] }); //ix = i + (i - 1) * (s - 1)
			pool_avg_r(y(), ix, ks, 1, ks - pa - 1);
			return next->predict();
		}
		dim1<3> din;
		shp2 ks, st, pa;
	};
}