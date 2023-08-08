#pragma once
#include "Layer.h"
namespace BNN {
	class Dense : public Layer {
	public:
		Dense(){}
		Dense(shp2 d, Afun af = Afun::t_lrelu) : Dense(d, nullptr, af) {}
		Dense(shp2 d, Layer* prev, Afun af = Afun::t_lrelu) :
			Layer({ 1,d[0],1 }, prev), dz(odims()), b(odims()),
			w(1, d[1], prev ? psize() : d[0]), af(af) {
			_init();
		}
		void init() override { _init(); }
		void derivative(bool ptrain) override {
			dz = y() * dz;
			if(ptrain) mul_r(x().reshape(dim1<3>{ 1, wdim(2), 1 }), w, dz, {0,0});
		}
		void gradient(Tensor& dw, Tensor& db, bool ptrain, float inv_n = 1.f) override {
			dz = y() * dz;
			db += dz * inv_n;
			dw += mul(dz, x().reshape(dim1<3>{ 1, 1, wdim(2) }), { 1,0 })* inv_n;
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
			shp2 d(1,1); int af;
			in >> d[0] >> d[1] >> af; in.ignore(1, '\n');
			auto tmp = new Dense({d[0],d[1]}, nullptr, (Afun::Type)af);
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
			b = b.random() * 0.5f - 0.25f;
			w = w.random() * 0.5f - 0.25f;
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute(fma(w, x.reshape(dim1<3>{ 1, wdim(2), 1 }), b).unaryExpr(af.fx()));
		}
		Tensor comp_dyn(const Tensor& x) const override {
			return next->comp_dyn(fma(w.broadcast(dim1<3>{1, 1, x.size() / wdim(2)}), x, b).unaryExpr(af.fx()));
		}
		const Tensor& predict() override {
			fma_r(y(), w, x().reshape(dim1<3>{ 1, wdim(2), 1 }), b);
			dz = y().unaryExpr(af.dx());
			y() = y().unaryExpr(af.fx());
			return next->predict();
		}
		Tensor dz, b;
		Tensor w;
		Afun af;
	};
}