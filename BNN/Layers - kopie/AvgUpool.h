#pragma once
#include "Layer.h"
namespace BNN {
	//Can be used for either unpooling or upsampling
	class AvgUpool : public Layer {
	public:
		AvgUpool() {}
		AvgUpool(shp3 din, shp2 ks, shp2 st, shp2 pa, Layer* prev = nullptr) :
			Layer({ din[0],t_dim(din[1],ks[0],st[0],pa[0]),t_dim(din[2],ks[1],st[1],pa[1]) }, prev),
			w(1, ks[0], ks[1]), din(din), ks(ks), st(st), pa(pa) {
			_init();
		}
		AvgUpool(shp2 ks, shp2 st, shp2 pa, Layer* prev) :
			Layer({ prev->odim(0),t_dim(prev->odim(1),ks[0],st[0],pa[0]),t_dim(prev->odim(2),ks[1],st[1],pa[1]) }, prev),
			w(1, ks[0], ks[1]), din(pdims()), ks(ks), st(st), pa(pa) {
			_init();
		}
		void derivative(bool ptrain) override {
			w.setConstant(1.f / w.size());
			if(ptrain) all_convolve(x(), y(), w, st, pa);
		}
		void print()const override {
			println("AvgUpl\t|", "\tDim:", odim(0), odim(1), odim(2), "\tKernel:", ks[0], ks[1], "\tStride:", st[0], st[1], "\tPad:", pa[0], pa[1]);
		}
		void save(std::ostream& out)const override {
			out << "Hidden AvgUpool" SPC din[0] SPC din[1] SPC din[2] SPC ks[0]
				SPC ks[1] SPC st[0] SPC st[1] SPC pa[0] SPC pa[1] << "\n";
			next->save(out);
		}
		static auto load(std::istream& in) {
			shp3 d; shp2 k, s, p;
			in >> d[0] >> d[1] >> d[2] >> k[0]
				>> k[1] >> s[0] >> s[1] >> p[0] >> p[1];
			return new AvgUpool(d, k, s, p);
		}
		AvgUpool* clone() const override { return new AvgUpool(*this); }
		dim1<3> idims() const override { return din; }
		LType type() const override { return t_AvgUpool; }
	private:
		void init()override { _init(); }
		void _init() {
			w.setConstant(float(st[0] * st[1]) / w.size());
		}
		Tensor compute(const Tensor& x) const override {
			auto ix = x.inflate(dim1<3>{ 1, st[0], st[1] });
			return next->compute(aconv(ix, w.constant(float(st[0] * st[1]) / w.size()), 1, ks - pa - 1));
		}
		Tensor compute_ds(const Tensor& x) const override {
			auto ix = x.inflate(dim1<3>{ 1, st[0], st[1] });
			return next->compute_ds(aconv(ix, w.constant(float(st[0] * st[1]) / w.size()), 1, ks - pa - 1));
		}
		const Tensor& predict() override {
			w.setConstant(float(st[0] * st[1]) / w.size());
			if(st[0] > 1 || st[1] > 1) {
				auto ix = x().inflate(dim1<3>{ 1, st[0], st[1] });
				all_convolve(y(), ix, w, 1, ks - pa - 1);
			}
			else all_convolve(y(), x(), w, 1, ks - pa - 1);
			return next->predict();
		}
		const Tensor& predict(const Tensor& x) override {
			w.setConstant(float(st[0] * st[1]) / w.size());
			if(st[0] > 1 || st[1] > 1) {
				auto ix = x.inflate(dim1<3>{ 1, st[0], st[1] }); //ix = i + (i - 1) * (s - 1)
				all_convolve(y(), ix, w, 1, ks - pa - 1);
			}
			else all_convolve(y(), x, w, 1, ks - pa - 1);
			return next->predict(y());
		}
		Tensor w;
		dim1<3> din;
		shp2 ks, st, pa;
	};
}