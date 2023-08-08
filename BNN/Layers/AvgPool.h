#pragma once
#include "Layer.h"
namespace BNN {
	//Can be used for either pooling or downsampling
	class AvgPool : public Layer {
	public:
		AvgPool() {}
		AvgPool(shp3 din, shp2 ks, shp2 st, shp2 pa, Layer* prev = nullptr) :
			Layer({ din[0],c_dim(din[1],ks[0],st[0],pa[0]),c_dim(din[2],ks[1],st[1],pa[1]) }, prev),
			w(1, ks[0], ks[1]), din(din), ks(ks), st(st), pa(pa) {
			_init();
		}
		AvgPool(shp2 ks, shp2 st, shp2 pa, Layer* prev) :
			Layer({ prev->odim(0),c_dim(prev->odim(1),ks[0],st[0],pa[0]),c_dim(prev->odim(2),ks[1],st[1],pa[1]) }, prev),
			w(1, ks[0], ks[1]), din(pdims()), ks(ks), st(st), pa(pa) {
			_init();
		}
		void derivative() override {
			w.setConstant(float(st[0] * st[1]) / w.size());
			auto dy = y().inflate(dim1<3>{ 1, st[0], st[1] });
			//much faster than pooling...
			if(ptype() != t_Input) aconv_r(x().reshape(din), dy, w, 1, ks - pa - 1);
		}
		void print()const override {
			println("AvgPl\t|", "\tIn:", din[0], din[1], din[2],
				"\tOut:", odim(0), odim(1), odim(2), "\tKernel:", ks[0], ks[1], "\tStride:", st[0], st[1], "\tPad:", pa[0], pa[1]);
		}
		void save(std::ostream& out)const override {
			out << "Hidden AvgPool" SPC din[0] SPC din[1] SPC din[2] SPC ks[0]
				SPC ks[1] SPC st[0] SPC st[1] SPC pa[0] SPC pa[1] << "\n";
			next->save(out);
		}
		static auto load(std::istream& in) {
			shp3 d; shp2 k, s, p;
			in >> d[0] >> d[1] >> d[2] >> k[0]
				>> k[1] >> s[0] >> s[1] >> p[0] >> p[1];
			return new AvgPool(d, k, s, p);
		}
		AvgPool* clone() const override { return new AvgPool(*this); }
		dim1<3> idims() const override { return din; }
		LType type() const override { return t_AvgPool; }
	private:
		void init()override { _init(); }
		void _init() {
			w.setConstant(1.f / w.size());
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute(aconv(x.reshape(din), w.constant(1.f / w.size()), st, pa));
		}
		Tensor comp_dyn(const Tensor& x) const override {
			return next->comp_dyn(aconv(x, w.constant(1.f / w.size()), st, pa));
		}
		const Tensor& predict() override {
			w.setConstant(1.f / w.size());
			aconv_r(y(), x().reshape(din), w, st, pa);
			return next->predict();
		}
		Tensor w;
		dim1<3> din;
		shp2 ks, st, pa;
	};
}