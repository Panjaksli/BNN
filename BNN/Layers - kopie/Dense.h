#pragma once
#include "Layer.h"
namespace BNN {
	class Dense : public Layer {
	public:
		Dense() {}
		Dense(shp2 d, Afun af = Afun::t_lrelu) : Dense(d, nullptr, af) {}
		Dense(shp2 d, Layer* prev, Afun af = Afun::t_lrelu) :
			Layer({ d[0],1,1 }, prev), dz(odims()), b(odims()),
			w(d[1], prev ? psize() : d[0], 1), af(af) {
			_init();
		}
		void init() override { _init(); }
		void derivative(bool ptrain) override {
			dz = y() * dz.unaryExpr(af.dx());
			if(ptrain) mul_r(x(), w, dz, { 0,0 });
		}
		void gradient(Tensor& dw, Tensor& db, bool ptrain) override {
			dz = y() * dz.unaryExpr(af.dx());
			db += dz;
			acc_mul(dw, dz, x(), { 1,0 });
			if(ptrain) mul_r(x(), w, dz, { 0,0 });
		}
		void print()const override {
			println("Dense\t|", "\tDim:", 1, 1, odim(1), "\tAf:", af.name());
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
		dim1<3> idims() const override { return dim1<3>{wdim(0), 1, 1}; }
		LType type() const override { return t_Dense; }
	private:
		void _init() {
			b.setZero();//squared_init(b, 0.2f);
			xavier_init(w, w.dimension(0), w.dimension(1));
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute(fma(w, x, b).unaryExpr(af.fx()));
		}
		//this function exists, but should not be used... 
		Tensor compute_ds(const Tensor& x) const override {
			return next->compute_ds(fma(w.broadcast(dim1<3>{x.size() / wdim(2),1,1}), x, b).unaryExpr(af.fx()));
		}
		const Tensor& predict() override {
			fma_r(dz, w, x(), b);
			y() = dz.unaryExpr(af.fx());
			return next->predict();
		}
		const Tensor& predict(const Tensor& x) override {
			fma_r(dz, w, x, b);
			return next->predict(y() = dz.unaryExpr(af.fx()));
		}
		Tensor dz, b;
		Tensor w;
		Afun af;
	};
}