#pragma once
#include "Layer.h"
namespace BNN {
	//Separable Convolution
	class SConv : public Layer {
	public:
		SConv() {}
		SConv(shp3 din, shp2 ks, shp2 st, shp2 pa, Layer* prev = nullptr) :
			Layer({ din[0], c_dim(din[1],ks[0],st[0],pa[0]), c_dim(din[2],ks[1],st[1],pa[1]) }, prev),
			w(din[0], ks[0], ks[1]), b(0, 0, 0),
			din(din), ks(ks), st(st), pa(pa) {
			_init();
		}
		SConv(shp2 ks, shp2 st, shp2 pa, Layer* prev) :
			Layer({ prev->odim(0), c_dim(prev->odim(1),ks[0],st[0],pa[0]), c_dim(prev->odim(2),ks[1],st[1],pa[1]) }, prev),
			w(pdim(0), ks[0], ks[1]), b(0, 0, 0),
			din(pdims()), ks(ks), st(st), pa(pa) {
			_init();
		}
		void init() override { _init(); }
		void derivative(bool ptrain) override {
			if(st[0] > 1 || st[1] > 1) {
				auto dy = y().inflate(dim1<3>{ 1, st[0], st[1] });
				if(ptrain) rev_convolve_1to1(x(), dy, w, 1, ks - pa - 1);
			}
			else if(ptrain) rev_convolve_1to1(x(), y(), w, 1, ks - pa - 1);
		}
		void gradient(Tensor& dw, Tensor& db, bool ptrain) override {
			if(st[0] > 1 || st[1] > 1) {
				auto dy = y().inflate(dim1<3>{ 1, st[0], st[1] });
				acc_convolve_1to1(dw, x(), dy, 1, pa);
				if(ptrain) rev_convolve_1to1(x(), dy, w, 1, ks - pa - 1);
			}
			else {
				acc_convolve_1to1(dw, x(), y(), 1, pa);
				if(ptrain) rev_convolve_1to1(x(), y(), w, 1, ks - pa - 1);
			}
			
		}

		void print()const override {
			println("SConv\t|", "\tDim:", odim(0), odim(1), odim(2), "\tKernel:", ks[0], ks[1],
				"\tStride:", st[0], st[1], "\tPad:", pa[0], pa[1]);
		}
		void save(std::ostream& out)const override {
			out << "Hidden SConv" SPC din[0] SPC din[1] SPC din[2] SPC ks[0]
				SPC ks[1] SPC st[0] SPC st[1] SPC pa[0] SPC pa[1] << "\n";
			out.write((const char*)w.data(), 4 * w.size());
			out << "\n";
			next->save(out);
		}
		static auto load(std::istream& in) {
			shp3 d; shp2 k, s, p;
			in >> d[0] >> d[1] >> d[2] >> k[0]
				>> k[1] >> s[0] >> s[1] >> p[0] >> p[1]; in.ignore(1, '\n');
			auto tmp = new SConv(d, k, s, p);
			in.read((char*)tmp->get_w()->data(), tmp->w.size() * 4);
			return tmp;
		}
		SConv* clone() const override { return new SConv(*this); }
		Tensor* get_w() override { return &w; }
		Tensor* get_b() override { return &b; }
		dim1<3> wdims() const override { return w.dimensions(); }
		dim1<3> bdims() const override { return b.dimensions(); }
		dim1<3> idims() const override { return din; }
		LType type() const override { return t_SConv; }
	private:
		void _init() {
			xavier_init(w, ks[0] * ks[1], ks[0] * ks[1]);
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute(conv_1to1(x, w, st, pa));
		}
		Tensor compute_ds(const Tensor& x) const override {
			return next->compute_ds(conv_1to1(x, w, st, pa));
		}
		const Tensor& predict() override {
			convolve_1to1(y(), x(), w, st, pa);
			return next->predict();
		}
		const Tensor& predict(const Tensor& x) override {
			convolve_1to1(y(), x, w, st, pa);
			return next->predict(y());
		}
		Tensor w, b;
		dim1<3> din;
		shp2 ks, st, pa;
	};
}