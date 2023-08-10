#pragma once
#include "Layer.h"
namespace BNN {
	//OutShuf layer, allows shuffling multiple channels to rgb image, provides error calculation (output gradient)
	//Eigen::Tensor<float, 3> y = x.reshape(dim1<5>{c / r2, r, r, h, w}).shuffle(dim1<5>{0, 2, 3, 1, 4}).reshape(dim1<3>{c / r2, h* r, w* r});
	//Eigen::Tensor<float, 3> z = x - y.reshape(dim1<5>{c / r2, r, h, r, w}).shuffle(dim1<5>{0, 3, 1, 2, 4}).reshape(x.dimensions());
	class OutShuf : public Layer {
	public:
		OutShuf() {}
		OutShuf(shp3 d, int r = 1, Layer* prev = nullptr, Afun af = Afun::t_lin, Efun ef = Efun::t_mse) :
			Layer({ d[0] / (r * r),r * d[1],r * d[2] }, prev), ef(ef), af(af), r(r) {}
		OutShuf(Layer* prev, int r = 1, Afun af = Afun::t_lin, Efun ef = Efun::t_mse) :
			Layer({ prev->odim(0) / (r * r),r * prev->odim(1),r * prev->odim(2) }, prev), ef(ef), af(af), r(r) {}
		const Tensor& output() const { return y(); }
		void derivative(bool ptrain) override {
			if(ptrain) x() = y().reshape(dim1<5>{odim(0), r, idim(1), r, idim(2)})
				.shuffle(dim1<5>{0, 3, 1, 2, 4})
				.reshape(pdims());
		}
		float error(const Tensor& y0) override {
			float cost = 0;
			cost = fsca(y().binaryExpr(y0, ef.fx()).mean()).coeff();
			x() = y().binaryExpr(y0, ef.dx())
				.reshape(dim1<5>{odim(0), r, idim(1), r, idim(2)})
				.shuffle(dim1<5>{0, 3, 1, 2, 4})
				.reshape(pdims());
			return cost;
		}
		void print()const override {
			println("OutShuf\t|", "\tIn:", idim(0), idim(1), idim(2), "\tOut:", odim(0), odim(1), odim(2));
		}
		void save(std::ostream& out)const override {
			out << "Hidden OutShuf" SPC idim(0) SPC idim(1) SPC idim(2) SPC af.type SPC ef.type SPC r << "\n";
		}
		dim1<3> idims() const override { return dim1<3>{odim(0)* r* r, odim(1) / r, odim(2) / r}; }
		static auto load(std::istream& in) {
			shp3 d; int af; int ef; int r;
			in >> d[0] >> d[1] >> d[2] >> af >> ef >> r;
			return new OutShuf(d, r, nullptr, (Afun::Type)af, (Efun::Type)ef);
		}
		OutShuf* clone() const override { return new OutShuf(*this); }
		LType type() const override { return t_OutShuf; }
	private:
		Tensor compute(const Tensor& x) const override {
			return x.reshape(dim1<5>{odim(0), r, r, idim(1), idim(2)})
				.shuffle(dim1<5>{0, 2, 3, 1, 4})
				.reshape(odims())
				.unaryExpr(af.fx());
		}
		Tensor comp_dyn(const Tensor& x) const override {
			return x.reshape(dim1<5>{odim(0), r, r, idim(1), idim(2)})
				.shuffle(dim1<5>{0, 2, 3, 1, 4})
				.reshape(odims())
				.unaryExpr(af.fx());
		}
		const Tensor& predict(const Tensor& x) override {
			return y() = x.reshape(dim1<5>{odim(0), r, r, idim(1), idim(2)})
				.shuffle(dim1<5>{0, 2, 3, 1, 4})
				.reshape(odims())
				.unaryExpr(af.fx());
		}
		const Tensor& predict() override {
			return y() = x().reshape(dim1<5>{odim(0), r, r, idim(1), idim(2)})
				.shuffle(dim1<5>{0, 2, 3, 1, 4})
				.reshape(odims())
				.unaryExpr(af.fx());
		}
		Efun ef;
		Afun af;
		int r = 2;
	};
}