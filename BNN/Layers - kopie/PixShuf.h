#pragma once
#include "Layer.h"
namespace BNN {
	//PixShuf layer, allows shuffling multiple channels to rgb image, provides error calculation (output gradient)
	class PixShuf : public Layer {
	public:
		PixShuf() {}
		PixShuf(shp3 d, idx r = 1, Layer* prev = nullptr) :
			Layer({ d[0] / (r * r),r * d[1],r * d[2] }, prev), r(r) {}
		PixShuf(Layer* prev, idx r = 1) :
			Layer({ prev->odim(0) / (r * r),r * prev->odim(1),r * prev->odim(2) }, prev), r(r) {}
		const Tensor& output() const { return y(); }
		void derivative(bool ptrain) override {
			if(ptrain)
				x() = y().reshape(dim1<5>{odim(0), r, idim(1), r, idim(2)})
				.shuffle(dim1<5>{0, 3, 1, 2, 4})
				.reshape(pdims());
		}
		void print()const override {
			println("PixShuf\t|", "\tDim:", odim(0), odim(1), odim(2));
		}
		void save(std::ostream& out)const override {
			out << "Hidden PixShuf" SPC idim(0) SPC idim(1) SPC idim(2) SPC r << "\n";
			next->save(out);
		}
		dim1<3> idims() const override { return dim1<3>{odim(0)* r* r, odim(1) / r, odim(2) / r}; }
		static auto load(std::istream& in) {
			shp3 d; idx r;
			in >> d[0] >> d[1] >> d[2] >> r;
			return new PixShuf(d, r, nullptr);
		}
		PixShuf* clone() const override { return new PixShuf(*this); }
		LType type() const override { return t_PixShuf; }
	private:
		Tensor compute(const Tensor& x) const override {
			return next->compute(x.reshape(dim1<5>{odim(0), r, r, idim(1), idim(2)})
				.shuffle(dim1<5>{0, 2, 3, 1, 4})
				.reshape(odims()));
		}
		Tensor compute_ds(const Tensor& x) const override {
			return next->compute_ds(x.reshape(dim1<5>{x.dimension(0) / (r * r), r, r, x.dimension(1), x.dimension(2)})
				.shuffle(dim1<5>{0, 2, 3, 1, 4})
				.reshape(dim1<3>{x.dimension(0) / (r * r), r* x.dimension(1), r* x.dimension(2)}));
		}
		const Tensor& predict(const Tensor& x) override {
			y() = x.reshape(dim1<5>{odim(0), r, r, idim(1), idim(2)})
				.shuffle(dim1<5>{0, 2, 3, 1, 4})
				.reshape(odims());
			return next->predict(y());
		}
		const Tensor& predict() override {
			y() = x().reshape(dim1<5>{odim(0), r, r, idim(1), idim(2)})
				.shuffle(dim1<5>{0, 2, 3, 1, 4})
				.reshape(odims());
			return next->predict();
		}
		idx r = 2;
	};
}