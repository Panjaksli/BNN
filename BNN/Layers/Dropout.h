#pragma once
#include "Layer.h"
namespace BNN {
	//Dropout layer, allows reshaping
	class Dropout : public Layer {
	public:
		Dropout(float rate, Layer* prev) : Layer(prev->odims(), prev), rate(rate) { _init(); }
		Dropout(float rate, shp3 din, Layer* prev = nullptr) : Layer(din, prev), rate(rate) { _init(); }
		void init() override { _init(); }
		void derivative(bool ptrain) override {
			float weight = (1.f / (1.f - rate));
			uint32_t sd = seed;
			if(ptrain) x() = y().reshape(pdims()).unaryExpr([&sd, weight, this](float x) { return x * (rand_fl(sd) > rate) * weight; });;
		}
		void print()const override {
			println("Dropout\t|", "\tDim:", odim(0), odim(1), odim(2), "\tRate:", rate);
		}
		void save(std::ostream& out)const override {
			out << "Hidden Dropout" SPC odim(0) SPC odim(1) SPC odim(2) SPC rate << "\n";
			next->save(out);
		}
		static auto load(std::istream& in) {
			shp3 d; float rate;
			in >> d[0] >> d[1] >> d[2] >> rate;
			return new Dropout(rate, d);
		}
		Dropout* clone() const override { return new Dropout(*this); }
		LType type() const override { return t_Dropout; }
	private:
		Tensor compute(const Tensor& x) const override {
			//We should skip the dropout during normal computation !
			return next->compute(x);
		}
		Tensor compute_ds(const Tensor& x) const override {
			return next->compute(x);
		}
		const Tensor& predict() override {
			float weight = (1.f / (1.f - rate));
			uint32_t sd = seed;
			y() = x().reshape(idims()).unaryExpr([&sd, weight, this](float x) { return x * (rand_fl(sd) > rate) * weight; });
			return next->predict();
		}
		const Tensor& predict(const Tensor& x) override {
			float weight = (1.f / (1.f - rate));
			uint32_t sd = seed;
			return next->predict(y() = x.reshape(idims()).unaryExpr([&sd, weight, this](float x) { return x * (rand_fl(sd) > rate) * weight; }));
		}
		void update() override { _init(); }
		void _init() {
			seed = xorshift32();
		}
		static uint32_t random(uint32_t& x) {
			x ^= x << 13;
			x ^= x >> 17;
			x ^= x << 5;
			return x;
		}
		static float rand_fl(uint32_t& x) {
			uint32_t y = 0x3f800000 | (random(x) & 0x007FFFFF);
			return *(float*)&y - 1.f;
		}
		float rate = 0.1f;
		uint32_t seed = 0;
	};
}