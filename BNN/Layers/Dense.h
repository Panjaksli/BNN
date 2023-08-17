#pragma once
#include "Layer.h"
namespace BNN {
	class Dense : public Layer {
	public:
		Dense() {}
		Dense(shp2 d, Afun af = Afun::t_lrelu) : Dense(d, nullptr, af) {}
		Dense(shp2 d, Layer* prev, Afun af = Afun::t_lrelu) :
			Layer({ 1,d[0],1 }, prev), dz(odims()), b(odims()),
			w(1, d[1], prev ? psize() : d[0]), af(af) {
			_init();
		}
		void init() override { _init(); }
		void derivative(bool ptrain) override {
			dz = y() * dz.unaryExpr(af.dx());
			if(ptrain) mul_r(x().reshape(dim1<3>{ 1, wdim(2), 1 }), w, dz, { 0,0 });
		}
		void gradient(Tensor& dw, Tensor& db, bool ptrain, float inv_n = 1.f) override {
			dz = y() * dz.unaryExpr(af.dx());
			db += dz * inv_n;
			acc_mul(dw, dz, x().reshape(dim1<3>{ 1, 1, wdim(2) }), inv_n, { 1,0 });
			if(ptrain) mul_r(x().reshape(dim1<3>{ 1, wdim(2), 1 }), w, dz, { 0,0 });
		}
		void print()const override {
			println("Dense\t|", "\tIn:", 1, wdim(2), 1, "\tOut:", odim(0), odim(1), odim(2));
		}
		void save(std::ostream& out)const override {
			out << "Hidden Dense" SPC wdim(2) SPC wdim(1) SPC af.type << "\n";
			out.write((const char*)w.data(), 4 * w.size());
			out.write((const char*)b.data(), 4 * b.size());
			out << "\n";
			next->save(out);
		}
		static auto load(std::istream& in) {
			shp2 d(1, 1); idx af;
			in >> d[0] >> d[1] >> af; in.ignore(1, '\n');
			auto tmp = new Dense({ d[0],d[1] }, nullptr, (Afun::Type)af);
			in.read((char*)tmp->get_w()->data(), tmp->w.size() * 4);
			in.read((char*)tmp->get_b()->data(), tmp->b.size() * 4);
			return tmp;
		}
		Dense* clone() const override { return new Dense(*this); }
		Tensor* get_w() override { return &w; }
		Tensor* get_b() override { return &b; }
		dim1<3> wdims() const override { return w.dimensions(); }
		dim1<3> bdims() const override { return b.dimensions(); }
		dim1<3> idims() const override { return dim1<3>{1, wdim(2), 1}; }
		LType type() const override { return t_Dense; }
	private:
		void _init() {
			b = b.random() * 0.2f - 0.1f;
			w = w.random() * 0.2f - 0.1f;
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute(fma(w, x.reshape(dim1<3>{ 1, wdim(2), 1 }), b).unaryExpr(af.fx()));
		}
		//this function exists, but should not be used... 
		Tensor compute_ds(const Tensor& x) const override {
			return next->compute_ds(fma(w.broadcast(dim1<3>{1, 1, x.size() / wdim(2)}), x.reshape(dim1<3>{1, x.size(), 1}), b).unaryExpr(af.fx()));
		}
		const Tensor& predict() override {
			fma_r(dz, w, x().reshape(dim1<3>{ 1, wdim(2), 1 }), b);
			y() = dz.unaryExpr(af.fx());
			return next->predict();
		}
		const Tensor& predict(const Tensor& x) override {
			fma_r(dz, w, x.reshape(dim1<3>{ 1, wdim(2), 1 }), b);
			return next->predict(y() = dz.unaryExpr(af.fx()));
		}
		Tensor dz, b;
		Tensor w;
		Afun af;
	};
}