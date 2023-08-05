#pragma once
#include "Layer.h"
namespace BNN {
	class TConv : public Layer {
	public:
		TConv(shp3 din, idx nch, shp2 ks, shp2 st, shp2 pa, Layer* prev = nullptr, Afun af = Afun::t_lrelu) :
			Layer({ nch,t_dim(din[1],ks[0],st[0],pa[0]),t_dim(din[2],ks[1],st[1],pa[1]) }, prev),
			dz(dim_y()), b(dim_y()), w(din[0] * nch, ks[0], ks[1]), din(din), ks(ks), st(st), pa(pa), af(af) {
			_init();
		}
		TConv(idx nch, shp2 ks, shp2 st, shp2 pa, Layer* prev, Afun af = Afun::t_lrelu) :
			Layer({ nch,t_dim(prev->dim_y(1),ks[0],st[0],pa[0]),t_dim(prev->dim_y(2),ks[1],st[1],pa[1]) }, prev),
			dz(dim_y()), b(dim_y()), w(dim_x(0)* nch, ks[0], ks[1]), din(prev->dim_y()), ks(ks), st(st), pa(pa), af(af) {
			_init();
		}
		void init() override { _init(); }
		void derivative() override {
			dz = y() * dz;
			auto wr = w.reverse(dimx<bool, 3>{false, true, true});
			idx p1 = t_pad(din[1], ks[0], st[0], dim_y(1));
			idx p2 = t_pad(din[2], ks[1], st[1], dim_y(2));
			conv_r(x().reshape(din), dz, wr, st, {p1, p2});
		}
		void gradient(Tensor& dw, Tensor& db, float inv_n = 1.f) override {
			dz = y() * dz;
			// eg0. i=2; k=2, s=2, p=0, o=4
			// eg1. i=3; k=3, s=2, p=1, o=5
			auto wr = w.reverse(dimx<bool, 3>{false, true, true});
			auto dx = x().reshape(din).inflate(dim1<3>{ 1, st[0], st[1] }); //ix = i + (i - 1) * (s - 1) = i * s + 1 - s
			//ix0 = 2 + 1 = 3
			//ix1 = 3 + 2 = 5
			idx pt1 = ti_pad(din[1], ks[0], st[0], dim_y(1));
			idx pt2 = ti_pad(din[2], ks[1], st[1], dim_y(2));
			db += dz * inv_n;
			dw += iconv(dx, dz, 1, { pt1,pt2 }) * inv_n; //k = ix - o + 2p + 1 -> p = (k - ix + o - 1) / 2 = (k + s * (1 - i) + o - 2) / 2 
			//pt0 = (2 - 2 + 4 - 2) / 2 = 1
			//pt1 = (3 - 5 + 5 - 1) / 2 = 1
			idx p1 = t_pad(din[1], ks[0], st[0], dim_y(1));
			idx p2 = t_pad(din[2], ks[1], st[1], dim_y(2));
			conv_r(x().reshape(din), dz, wr, st, { p1, p2 });  //i = 1 + (o - k + 2p) / s  -> p = ((i - 1) * s - o + k) / 2
			//p0 = ((2 - 1) * 2 - 4 + 2) / 2 = 0
			//p1 = ((3 - 1) * 2 - 5 + 3) / 2 = 1
		}
		Tensor* get_w() override { return &w; }
		Tensor* get_b() override { return &b; }
		dim1<3> dim_w() const override { return w.dimensions(); }
		dim1<3> dim_b() const override { return b.dimensions(); }
		idx dim_w(idx i) const override { return w.dimension(i); }
		idx dim_b(idx i) const override { return b.dimension(i); }
		idx sz_in() const override { return product(din); }
		dim1<3> dim_in() const override { return din; }
		void print()const override {
			println("TConv\t|", "\tIn:", din[0], din[1], din[2],
				"\tOut:", dim_y(0), dim_y(1), dim_y(2), "\tKernel:", dim_w(1), dim_w(2), "\tStride:", st[0], st[1], "\tPad:", pa[0], pa[1]);
		}
		TConv* clone() const override { return new TConv(*this); }
	private:
		void _init() {
			b = b.random() * 0.5f - 0.25f;
			w = w.random() * 0.5f - 0.25f;
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute((conv(x.reshape(din).inflate(dim1<3>{ 1, st[0], st[1] }), w, 1, ks - pa - 1) + b).unaryExpr(af.fx()));
		}
		const Tensor& predict() override {
			//eg0.  o=5; k=3, s=2, p=0, i=2
			//eg1.  o=5; k=3, s=2, p=1, i=3
			auto ix = x().reshape(din).inflate(dim1<3>{ 1, st[0], st[1] }); //ix = i + (i - 1) * (s - 1)
			//ix0 = 2 + 1 = 3
			//ix1 = 3 + 2 = 5
			conv_r(y(), ix, w, 1, ks - pa - 1); //y = 1 + (ix - k + 2*(k-p-1)) / 1 = ix + k - 2p - 1
			//y0 = 3 + 3 - 0 - 1 = 5 == o
			//y1 = 5 + 3 - 2 - 1 = 5 == o
			y() = y() + b;
			dz = y().unaryExpr(af.dx());
			y() = y().unaryExpr(af.fx());
			return next->predict();
		}
		Tensor dz, b;
		Tensor w;
		dim1<3> din;
		shp2 ks, st, pa;
		Afun af;
	};
}