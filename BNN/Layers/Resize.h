#pragma once
#include "Layer.h"
namespace BNN {
	class Resize : public Layer {
	public:
		Resize() {}
		Resize(shp3 d, double r1, double r2, Interpol filter, Layer* prev = nullptr) : Layer(shp3(d[0], d[1] * r1, d[2] * r2), prev), r1(r1), r2(r2), filter(filter) {}
		Resize(Layer* prev, double r, Interpol filter = Cubic) : Layer(shp3(prev->odim(0), r* prev->odim(1), r* prev->odim(2)), prev), r1(r), r2(r), filter(filter) {}
		Resize(Layer* prev, double r1, double r2, Interpol filter = Cubic) : Layer(shp3(prev->odim(0), r1* prev->odim(1), r2* prev->odim(2)), prev), r1(r1), r2(r2), filter(filter) {}
		const Tensor& output() const { return y(); }
		void derivative(bool ptrain) override {
			if(ptrain) resize_r({ x(), idims() }, y(), filter);
		}
		void print()const override {
			println("Resize\t|", "\tDim:", odim(0), odim(1), odim(2), "\tFilter:", to_cstr(filter));
		}
		void save(std::ostream& out)const override {
			out << "Hidden Resize" SPC idim(0) SPC idim(1) SPC idim(2) SPC r1 SPC r2 SPC filter << "\n";
			next->save(out);
		}
		dim1<3> idims() const override { return shp3(odim(0), odim(1) / r1, odim(2) / r2); }
		static auto load(std::istream& in) {
			shp3 d; double r1, r2; int fi;
			in >> d[0] >> d[1] >> d[2] >> r1 >> r2 >> fi;
			return new Resize(d, r1, r2, Interpol(fi), nullptr);
		}
		Resize* clone() const override { return new Resize(*this); }
		LType type() const override { return t_Resize; }
	private:
		Tensor compute(const Tensor& x) const override {
			return next->compute(resize(x, r1, r2, filter));
		}
		Tensor compute_ds(const Tensor& x) const override {
			return next->compute_ds(resize(x, r1, r2, filter));
		}
		const Tensor& predict(const Tensor& x) override {
			resize_r(y(), x.reshape(idims()), filter);
			return next->predict(y());
		}
		const Tensor& predict() override {
			resize_r(y(), x().reshape(idims()), filter);
			return next->predict();
		}
		double r1 = 1, r2 = 1;
		Interpol filter;
	};

}