#pragma once
#include "Eigen_util.h"
namespace BNN {
	class ten : public Tensor {
	public:
		ten() {}
		//A col vector
		ten(idx d1) : Tensor(1, d1, 1) { zero(); }
		//A matrix
		ten(idx d1, idx d2) : Tensor(1, d1, d2) { zero(); }
		//A cube
		ten(idx d1, idx d2, idx d3) : Tensor(d1, d2, d3) { zero(); }
		ten(dim1<3> d) : Tensor(d) { zero(); }
		ten(const Tensor& t) : Tensor(t) {}
		ten(Tensor&& t) : Tensor(std::move(t)) {}
		inline const float& operator[](int i) const {
			return data()[i];
		}
		inline float& operator[](int i) {
			return data()[i];
		}
		inline const idx dim(int i) const {
			return dimension(i);
		}
		inline const dim1<3> dims() const {
			return dim1<3>{dim(0), dim(1), dim(2)};
		}
		inline void zero() {
			setZero();
		}
		inline void test() {
			for (int i = 0; i < size(); i++) data()[i] = i;
		}
		inline void set(float x = 1.f) {
			setConstant(x);
		}
		inline void random(float min = 0.f, float max = 1.f) {
			setRandom();
			//*this = min + (max - min) * (*this);
		}
		//multiply all matrix combinations stored as a0b0,a0b1,a1b0,a1b1....
		inline ten& mul(const ten& a, const ten& b, pair dims = { 1, 0 }) {
			resize(a.dim(0) * b.dim(0), dims.first ? a.dim(1) : a.dim(2), dims.second ? b.dim(1) : b.dim(2));
			for (int i = 0; i < a.dim(0); i++) {
				for (int j = 0; j < b.dim(0); j++) {
					chip(i * b.dim(0) + j, 0) = a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{ dims });
				}
			}
			return *this;
		}
		//fma operation
		inline ten& fma(const ten& a, const ten& b, const ten& c, pair dims = { 1, 0 }) {
			resize(a.dim(0) * b.dim(0), dims.first ? a.dim(1) : a.dim(2), dims.second ? b.dim(1) : b.dim(2));
			for (int i = 0; i < a.dim(0); i++) {
				for (int j = 0; j < b.dim(0); j++) {
					chip(i * b.dim(0) + j, 0) = a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{ dims }) + c.chip(i * b.dim(0) + j, 0);
				}
			}
			return *this;
		}
		//multiply all matrix combinations and accumulate as a0b0 + a0b1, a1b0 + a1b1....
		inline ten& mul_acc(const ten& a, const ten& b, pair dims = { 1, 0 }) {
			resize(a.dim(0), dims.first ? a.dim(1) : a.dim(2), dims.second ? b.dim(1) : b.dim(2));
			for (int i = 0; i < a.dim(0); i++) {
				chip(i, 0).setZero();
				for (int j = 0; j < b.dim(0); j++) {
					chip(i, 0) += a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{dims});
				}
			}
			return *this;
		}
		//convolve multiple input channels with multiple filters !!!filters have to be stored in the same order as channels eg - 3 channels, 9 filters = 3 outputs!!!
		inline ten& conv(const ten& a, const ten& b, idx str = 1, idx pad = 0) {
			return conv(a, b, { str,str }, { pad,pad });
		}
		inline ten& conv(const ten& a, const ten& b, pair str, pair pad = { 0,0 }) {
			idx d0 = b.dim(0) / a.dim(0);
			idx d1 = conv_dim(a.dim(1), b.dim(1), str.first, pad.first);
			idx d2 = conv_dim(a.dim(2), b.dim(2), str.second, pad.second);
			resize(d0, d1, d2);
			dim1<2> st{str.first, str.second};
			dim2<2> pa{ pair{ pad.first, pad.first}, pair{ pad.second, pad.second }};
			for (int i = 0; i < d0; i++) {
				chip(i, 0).setZero();
				for (int j = 0; j < a.dim(0); j++) {
					chip(i, 0) += a.chip(j, 0).pad(pa).convolve(b.chip(i * a.dim(0) + j, 0), dim1<2>{0, 1}).stride(st);
				}
			}
			return *this;
		}
		inline ten& pool_max(const ten& a, idx ker = 2, idx str = 1) {
			return pool_max(a, { ker,ker }, { str,str });
		}
		inline ten& pool_max(const ten& a, pair ker, pair str = { 1,1 }) {
			idx d0 = a.dim(0);
			idx d1 = conv_dim(a.dim(1), ker.first, str.first, 0);
			idx d2 = conv_dim(a.dim(2), ker.second, str.second, 0);
			resize(d0, d1, d2);
			dim1<2> st{str.first, str.second};
			dim1<2> ks{ker.first, ker.second};
			for (int i = 0; i < d0; i++) {
				auto x = a.chip(i, 0);
				for (int k = 0; k < d2; k++) {
					for (int j = 0; j < d1; j++) {
						dim1<2> off{j* st[0], k* st[1]};
						fsca y = x.slice(off, ks).maximum();
						this->operator()(i, j, k) = y(0);
					}
				}
			}
			return *this;
		}
		inline ten& pool_avg(const ten& a, idx ker = 2, idx str = 1) {
			return pool_avg(a, { ker,ker }, { str,str });
		}
		inline ten& pool_avg(const ten& a, pair ker, pair str = { 1,1 }) {
			idx d0 = a.dim(0);
			idx d1 = conv_dim(a.dim(1), ker.first, str.first, 0);
			idx d2 = conv_dim(a.dim(2), ker.second, str.second, 0);
			resize(d0, d1, d2);
			dim1<2> st{str.first, str.second};
			dim1<2> ks{ker.first, ker.second};
			float mult = (1.f / (ks[0] * ks[1]));
			for (int i = 0; i < d0; i++) {
				auto x = a.chip(i, 0);
				for (int k = 0; k < d2; k++) {
					for (int j = 0; j < d1; j++) {
						dim1<2> off{j* st[0], k* st[1]};
						fsca y = x.slice(off, ks).sum();
						this->operator()(i, j, k) = y(0) * mult;
					}
				}
			}
			return *this;
		}
		void print() const {
			print_numpy(*this);
		}
	};
}