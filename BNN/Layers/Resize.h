#pragma once
#include "Layer.h"
namespace BNN {
	class Resize : public Layer {
	public:
		Resize() {}
		Resize(shp3 d, double r1, double r2, Layer* prev = nullptr) : Layer(shp3(d[0], d[1] * r1, d[2] * r2), prev), r1(r1), r2(r2) {}
		Resize(Layer* prev, double r) : Layer(shp3(prev->odim(0), r* prev->odim(1), r* prev->odim(2)), prev), r1(r), r2(r) {}
		Resize(Layer* prev, double r1, double r2) : Layer(shp3(prev->odim(0), r1* prev->odim(1), r2* prev->odim(2)), prev), r1(r1), r2(r2) {}
		const Tensor& output() const { return y(); }
		void derivative(bool ptrain) override {
			if(ptrain) bilinear({ x(), idims() }, y());
		}
		void print()const override {
			println("Resize\t|", "\tDim:", odim(0), odim(1), odim(2));
		}
		void save(std::ostream& out)const override {
			out << "Hidden Resize" SPC idim(0) SPC idim(1) SPC idim(2) SPC r1 SPC r2 << "\n";
			next->save(out);
		}
		dim1<3> idims() const override { return shp3(odim(0), odim(1) / r1, odim(2) / r2); }
		static auto load(std::istream& in) {
			shp3 d; double r1, r2;
			in >> d[0] >> d[1] >> d[2] >> r1 >> r2;
			return new Resize(d, r1, r2, nullptr);
		}
		Resize* clone() const override { return new Resize(*this); }
		LType type() const override { return t_Resize; }
	private:
		Tensor compute(const Tensor& x) const override {
			Tensor y(odims()); 
			bilinear(y, x.reshape(idims()));
			return next->compute(y);
		}
		Tensor compute_ds(const Tensor& x) const override {
			Tensor y(x.dimension(0), int(x.dimension(1) * r1), int(x.dimension(2) * r2)); 
			bilinear(y, x);
			return next->compute(y);
		}
		const Tensor& predict(const Tensor& x) override {
			bilinear(y(), x.reshape(idims()));
			return next->predict(y());
		}
		const Tensor& predict() override {
			bilinear(y(), x().reshape(idims()));
			return next->predict();
		}
		double r1 = 1, r2 = 1;
	};

}