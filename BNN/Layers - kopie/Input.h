#pragma once
#include "Layer.h"
namespace BNN {
	//Allows reshaping of the input
	class Input : public Layer {
	public:
		Input() {}
		Input(shp3 d) : Layer(d) {}
		void input(const Tensor& x) {
			y() = x.reshape(odims());
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute(x.reshape(odims()));
		}
		Tensor compute_ds(const Tensor& x) const override {
			return next->compute_ds(x);
		}
		const Tensor& predict(const Tensor& x) override {
			input(x);
			return next->predict();
		}
		const Tensor& predict() override {
			return next->predict();
		}
		void print()const override {
			println("Input\t|", "\tDim:", odim(0), odim(1), odim(2));
		}
		void save(std::ostream& out)const override {
			out << "Hidden Input" SPC odim(0) SPC odim(1) SPC odim(2) << "\n";
			next->save(out);
		}
		static auto load(std::istream& in) {
			shp3 d;
			in >> d[0] >> d[1] >> d[2];
			return new Input(d);
		}
		Input* clone() const override { return new Input(*this); }
		LType type() const override { return t_Input; }
	private:
	};
}