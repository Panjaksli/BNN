#pragma once
#include "Layer.h"
namespace BNN {
	//OutShuf layer, allows shuffling multiple channels to rgb image, provides error calculation (output gradient)
	class OutShuf : public Layer {
	public:
		OutShuf() {}
		OutShuf(shp3 d, idx r = 1, Layer* prev = nullptr, Efun ef = Efun::t_mse) :
			Layer({ d[0] / (r * r),r * d[1],r * d[2] }, prev), ef(ef),  r(r) {}
		OutShuf(Layer* prev, idx r = 1, Efun ef = Efun::t_mse) :
			Layer({ prev->odim(0) / (r * r),r * prev->odim(1),r * prev->odim(2) }, prev), ef(ef), r(r) {}
		const Tensor& output() const { return y(); }
		void derivative(bool ptrain) override {
			if(ptrain) 
				x() = y()
				.reshape(dim1<5>{odim(0), r, idim(1), r, idim(2)})
				.shuffle(dim1<5>{0, 3, 1, 2, 4})
				.reshape(pdims());
		}
		float error(const Tensor& y0) override {
			float cost = 0;
			cost = fsca(y().binaryExpr(y0, &Efun::mae::fx).mean()).coeff();
			x() = y().binaryExpr(y0, ef.dx())
				.reshape(dim1<5>{odim(0), r, idim(1), r, idim(2)})
				.shuffle(dim1<5>{0, 3, 1, 2, 4})
				.reshape(pdims());
			return cost;
		}
		void print()const override {
			println("OutShuf\t|", "\tDim:", odim(0), odim(1), odim(2), ef.name());
		}
		void save(std::ostream& out)const override {
			out << "Hidden OutShuf" SPC idim(0) SPC idim(1) SPC idim(2) SPC ef.type SPC r << "\n";
		}
		dim1<3> idims() const override { return dim1<3>{odim(0)* r* r, odim(1) / r, odim(2) / r}; }
		static auto load(std::istream& in) {
			shp3 d; idx ef; idx r;
			in >> d[0] >> d[1] >> d[2] >> ef >> r;
			return new OutShuf(d, r, nullptr,(Efun::Type)ef);
		}
		OutShuf* clone() const override { return new OutShuf(*this); }
		LType type() const override { return t_OutShuf; }
	private:
		Tensor compute(const Tensor& x) const override {
			return x.reshape(dim1<5>{odim(0), r, r, idim(1), idim(2)})
				.shuffle(dim1<5>{0, 2, 3, 1, 4})
				.reshape(odims());
		}
		Tensor compute_ds(const Tensor& x) const override {
			return x.reshape(dim1<5>{x.dimension(0) / (r * r), r, r, x.dimension(1), x.dimension(2)})
				.shuffle(dim1<5>{0, 2, 3, 1, 4})
				.reshape(dim1<3>{x.dimension(0) / (r * r), r* x.dimension(1), r* x.dimension(2)});
		}
		const Tensor& predict(const Tensor& x) override {
			return y() = x.reshape(dim1<5>{odim(0), r, r, idim(1), idim(2)})
				.shuffle(dim1<5>{0, 2, 3, 1, 4})
				.reshape(odims());
		}
		const Tensor& predict() override {
			return y() = x().reshape(dim1<5>{odim(0), r, r, idim(1), idim(2)})
				.shuffle(dim1<5>{0, 2, 3, 1, 4})
				.reshape(odims());
		}
		Efun ef;
		idx r = 2;
	};
}