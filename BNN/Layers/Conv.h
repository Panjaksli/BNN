#pragma once
#include "Layer.h"
namespace BNN {
	class Conv : public Layer {
	public:
		Conv(shp3 din, idx nch, shp2 ks, shp2 st, shp2 pa, Layer* prev  = nullptr, Afun af = Afun::t_lrelu) :
			Layer({ nch,c_dim(din[1],ks[0],st[0],pa[0]),c_dim(din[2],ks[1],st[1],pa[1]) }, prev),
			dz(dim_y()), b(dim_y()), w(din[0] * nch, ks[0], ks[1]), din(din), ks(ks), st(st), pa(pa), af(af) {
			_init();
		}
		Conv(idx nch, shp2 ks, shp2 st, shp2 pa, Layer* prev, Afun af = Afun::t_lrelu) :
			Layer({ nch,c_dim(prev->dim_y(1),ks[0],st[0],pa[0]),c_dim(prev->dim_y(2),ks[1],st[1],pa[1]) }, prev),
			dz(dim_y()), b(dim_y()), w(dim_x(0)* nch, ks[0], ks[1]), din(prev->dim_y()), ks(ks), st(st), pa(pa), af(af) {
			_init();
		}
		void init() override { _init(); }
		void derivative() override {
			dz = y() * dz;
			auto dy = dz.inflate(dim1<3>{ 1, st[0], st[1] });
			auto wr = w.reverse(dimx<bool, 3>{false, true, true});
			idx p1 = c_pad(din[1], ks[0], st[0], dim_y(1));
			idx p2 = c_pad(din[2], ks[1], st[1], dim_y(2));
			conv_r(x().reshape(din), dy, wr, 1, {p1, p2});
		}
		void gradient(Tensor& dw, Tensor& db, float inv_n = 1.f) override {
			dz = y() * dz;
			// eg0. i=5; k=3, s=2, p=0, o=2
			// eg1. i=5; k=3, s=2, p=1, o=3
			auto dy = dz.inflate(dim1<3>{ 1, st[0], st[1] }); //dy = o+(o-1)*(s-1)
			//dy0 = 2 + 1 = 3  
			//dy1 = 3 + (3 - 1) * (2 - 1) = 5
			auto wr = w.reverse(dimx<bool, 3>{false, true, true});
			db += dz * inv_n;
			//dw += iconv(x().reshape(din), dz, st, pa) * inv_n;
			dw += iconv(x().reshape(din), dy, 1, pa) * inv_n; //w = 1 + (i - dy + 2 * p) / 1 
			//w0 = 1 + (5 - 3 + 2 * 0) = 3 
			//w1 = 1 + (5 - 5 + 2) = 3
			//w0 = w1 !!!!!!!!
			idx p1 = c_pad(din[1], ks[0], st[0], dim_y(1));
			idx p2 = c_pad(din[2], ks[1], st[1], dim_y(2));
			conv_r(x().reshape(din), dy, wr, 1, { p1, p2 }); //i = 1 + (dy - k + 2p) / 1 ---> p = (i - 1 - dy + k) / 2
			//p0 = (5 - 1 - 3 + 3) / 2 = 2
			//p1 = (5 - 1 - 3 + 3) / 2 = 2
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
			println("Conv\t|", "\tIn:", din[0], din[1], din[2],
				"\tOut:", dim_y(0), dim_y(1), dim_y(2), "\tKernel:", dim_w(1), dim_w(2), "\tStride:", st[0], st[1], "\tPad:", pa[0], pa[1]);
		}
		Conv* clone() const override { return new Conv(*this);}
	private:
		void _init() {
			b = b.random() * 0.5f - 0.25f;
			w = w.random() * 0.5f - 0.25f;
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute((conv(x.reshape(din), w, st, pa) + b).unaryExpr(af.fx()));
		}
		const Tensor& predict() override {
			conv_r(y(), x().reshape(din), w, st, pa);
			y() = y() + b;
			dz = y().unaryExpr(af.dx());
			y() = y().unaryExpr(af.fx());
			return next->predict();
		}
		Tensor dz, b, w;
		dim1<3> din;
		shp2 ks, st, pa;
		Afun af;
	};
}