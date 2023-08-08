#pragma once
#include "Layer.h"
namespace BNN {
	//Allows reshaping of the input
	class Input : public Layer {
	public:
		Input(){}
		Input(shp3 d) : Layer(d) {}
		void input(const Tensor& x) {
			y() = x.reshape(odims());
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute(x.reshape(odims()));
		}
		Tenarr compute_batch(const Tenarr& x) const {
			dim1<3> d = last()->odims();
			Tenarr y(x.dimension(0), d[0], d[1], d[2]);
#pragma omp parallel for
			for (int i = 0; i < x.dimension(0); i++) {
				y.chip(i, 0) = compute(x.chip(i, 0));
			}
			return y;
		}
		const Tensor& predict(const Tensor& x) {
			input(x);
			return next->predict();
		}
		Tenarr predict_batch(const Tenarr& x) {
			dim1<3> d = last()->odims();
			Tenarr y(x.dimension(0), d[0], d[1], d[2]);
			for (int i = 0; i < x.dimension(0); i++) {
				y.chip(i, 0) = predict(x.chip(i, 0));
			}
			return y;
		}
		bool compile(Layer* pnode, Layer* nnode) override {
			set_prev(nullptr);
			set_next(nnode);
			return true;
		}
		void print()const override {
			println("Input\t|", "\tIn:", odim(0), odim(1), odim(2), "\tOut:", odim(0), odim(1), odim(2));
		}
		void save(std::ostream& out)const override {
			out << "Input" SPC odim(0) SPC odim(1) SPC odim(2) << "\n";
			next->save(out);
		}
		static auto load(std::istream& in) {
			shp3 d;
			in >> d[0] >> d[1] >> d[2];
			return new Input(d);
		}
		Input* clone() const override { return new Input(*this); }
		bool is_input() const override { return true; }
	private:
		const Tensor& predict() override { return next->predict(); }
	};
}