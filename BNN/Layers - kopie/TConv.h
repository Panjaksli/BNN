#pragma once
#include "Layer.h"
namespace BNN {
	class TConv : public Layer {
	public:
		TConv() {}
		TConv(shp3 din, idx nch, shp2 ks, shp2 st, shp2 pa, Layer* prev = nullptr, bool bias = true, Afun af = Afun::t_lrelu) :
			Layer({ nch,t_dim(din[1],ks[0],st[0],pa[0]),t_dim(din[2],ks[1],st[1],pa[1]) }, prev),
			dz(odims()), b(bias ? dim1<3>{nch, 1, 1} : dim1<3>{ 0,0,0 }), w(din[0] * nch, ks[0], ks[1]), din(din), ks(ks), st(st), pa(pa), af(af), bias(bias) {
			_init();
		}
		TConv(idx nch, shp2 ks, shp2 st, shp2 pa, Layer* prev, bool bias = true, Afun af = Afun::t_lrelu) :
			Layer({ nch,t_dim(prev->odim(1),ks[0],st[0],pa[0]),t_dim(prev->odim(2),ks[1],st[1],pa[1]) }, prev),
			dz(odims()), b(bias ? dim1<3>{nch, 1, 1} : dim1<3>{ 0,0,0 }), w(pdim(0)* nch, ks[0], ks[1]), din(pdims()), ks(ks), st(st), pa(pa), af(af), bias(bias) {
			_init();
		}
		void init() override { _init(); }
		void derivative(bool ptrain) override {
			dz = y() * dz.unaryExpr(af.dx());
			if(ptrain) rev_convolve(x(), dz, w, st, pa);
		}
		void gradient(Tensor& dw, Tensor& db, bool ptrain) override {
			dz = y() * dz.unaryExpr(af.dx());
			if(bias)db += dz.sum(dim1<2>{1, 2});
			if(st[0] > 1 || st[1] > 1) {
				auto dx = x().inflate(dim1<3>{ 1, st[0], st[1] });
				acc_convolve(dw, dx, dz, 1, ks - pa - 1);
			}
			else acc_convolve(dw, x(), dz, 1, ks - pa - 1);
			if(ptrain) rev_convolve(x(), dz, w, st, pa);
		}
		void print()const override {
			println("TConv\t|", "\tDim:", odim(0), odim(1), odim(2), "\tKernel:", ks[0], ks[1],
				"\tStride:", st[0], st[1], "\tPad:", pa[0], pa[1], "\tBias", bias, "\tAf:", af.name());
		}
		TConv* clone() const override { return new TConv(*this); }
		LType type() const override { return t_TConv; }
		void save(std::ostream& out)const override {
			out << "Hidden TConv" SPC din[0] SPC din[1] SPC din[2] SPC odim(0) SPC ks[0]
				SPC ks[1] SPC st[0] SPC st[1] SPC pa[0] SPC pa[1] SPC bias SPC af.type << "\n";
			out.write((const char*)w.data(), 4 * w.size());
			if(bias)out.write((const char*)b.data(), 4 * b.size());
			out << "\n";
			next->save(out);
		}
		static auto load(std::istream& in) {
			shp3 d; shp2 k, s, p; idx n; idx af; bool b;
			in >> d[0] >> d[1] >> d[2] >> n >> k[0]
				>> k[1] >> s[0] >> s[1] >> p[0] >> p[1] >> b >> af; in.ignore(1, '\n');
			auto tmp = new TConv(d, n, k, s, p, nullptr, b, Afun::Type(af));
			in.read((char*)tmp->get_w()->data(), tmp->w.size() * 4);
			if(b)in.read((char*)tmp->get_b()->data(), tmp->b.size() * 4);
			return tmp;
		}
		Tensor* get_w() override { return &w; }
		Tensor* get_b() override { return &b; }
		dim1<3> wdims() const override { return w.dimensions(); }
		dim1<3> bdims() const override { return b.dimensions(); }
		dim1<3> idims() const override { return din; }
	private:
		void _init() {
			if(bias)b.setZero();
			xavier_init(w, din[0] * ks[0] * ks[1], ks[0] * ks[1] * w.dimension(0) / din[0]);
		}
		Tensor compute(const Tensor& x) const override {
			if(st[0] > 1 || st[1] > 1) {
				if(bias)return next->compute((conv(x.inflate(dim1<3>{ 1, st[0], st[1] }), w, 1, ks - pa - 1) + b.broadcast(dim1<3>{1, odim(1), odim(2)})).unaryExpr(af.fx()));
				else return next->compute((conv(x.inflate(dim1<3>{ 1, st[0], st[1] }), w, 1, ks - pa - 1)).unaryExpr(af.fx()));
			}
			else {
				if(bias)return next->compute((conv(x, w, 1, ks - pa - 1) + b.broadcast(dim1<3>{1, odim(1), odim(2)})).unaryExpr(af.fx()));
				else return next->compute((conv(x, w, 1, ks - pa - 1)).unaryExpr(af.fx()));
			}
		}
		Tensor compute_ds(const Tensor& x) const override {
			if(st[0] > 1 || st[1] > 1) {
				if(bias) return next->compute_ds((conv(x.inflate(dim1<3>{ 1, st[0], st[1] }), w, 1, ks - pa - 1)
					+ b.broadcast(dim1<3>{1, t_dim(x.dimension(1), ks[0], st[0], pa[0]),
					t_dim(x.dimension(2), ks[1], st[1], pa[1])})).unaryExpr(af.fx()));
				else return next->compute_ds(conv(x.inflate(dim1<3>{ 1, st[0], st[1] }), w, 1, ks - pa - 1).unaryExpr(af.fx()));
			}
			else {
				if(bias) return next->compute_ds((conv(x, w, 1, ks - pa - 1)
					+ b.broadcast(dim1<3>{1, t_dim(x.dimension(1), ks[0], st[0], pa[0]),
					t_dim(x.dimension(2), ks[1], st[1], pa[1])})).unaryExpr(af.fx()));
				else return next->compute_ds(conv(x, w, 1, ks - pa - 1).unaryExpr(af.fx()));
			}
		}
		const Tensor& predict(const Tensor& x) override {
			auto ix = x.inflate(dim1<3>{ 1, st[0], st[1] });
			if(st[0] > 1 || st[1] > 1) convolve(dz, ix, w, 1, ks - pa - 1);
			else convolve(dz, x, w, 1, ks - pa - 1);
			if(bias)dz = dz + b.broadcast(dim1<3>{1, odim(1), odim(2)});
			return next->predict(y() = dz.unaryExpr(af.fx()));
		}
		const Tensor& predict() override {
			auto ix = x().inflate(dim1<3>{ 1, st[0], st[1] });
			if(st[0] > 1 || st[1] > 1) convolve(dz, ix, w, 1, ks - pa - 1);
			else convolve(dz, x(), w, 1, ks - pa - 1);
			if(bias)dz = dz + b.broadcast(dim1<3>{1, odim(1), odim(2)});
			y() = dz.unaryExpr(af.fx());
			return next->predict();
		}
		Tensor dz, b;
		Tensor w;
		dim1<3> din;
		shp2 ks, st, pa;
		Afun af;
		bool bias = true;
	};
}