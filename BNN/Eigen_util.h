#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include "Misc.h"

namespace BNN {
	using Tensor = Eigen::Tensor<float, 3>;
	using Tenarr = Eigen::Tensor<float, 4>;
	using fsca = Eigen::TensorFixedSize<float, Eigen::Sizes<>>;
	using idx = Eigen::Index;
	//using shp2 = Eigen::IndexPair<idx>;
	struct shp1 {
		shp1() {}
		shp1(idx first) : first(first) {}
		const idx& operator[](idx i) const { return i ? first : first; }
		idx& operator[](idx i) { return i ? first : first; }
		operator auto() {
			return std::array<idx, 3>{1, first, 1};
		}
		idx first;
	};
	struct shp2 {
		shp2() {}
		shp2(idx both) : first(both), second(both) {}
		shp2(idx first, idx second) : first(first), second(second) {}
		const idx& operator[](idx i) const { return i ? second : first; }
		idx& operator[](idx i) { return i ? second : first; }
		operator auto() {
			return std::array<idx, 3>{1, first, second};
		}
		friend shp2 operator-(shp2 x) { return shp2{ -x[0],-x[1] };}
		friend shp2 operator+(shp2 x, shp2 y) { return shp2{ x[0] + y[0],x[1] + y[1] }; }
		friend shp2 operator-(shp2 x, shp2 y) {return shp2{ x[0] - y[0],x[1] - y[1] };}
		idx first;
		idx second;
	};
	template <size_t N>
	using dim1 = Eigen::array<idx, N>;
	template <size_t N>
	using dim2 = Eigen::array<shp2, N>;
	template <class T, size_t N>
	using dimx = Eigen::array<T, N>;
	struct shp3 {
		shp3() {}
		shp3(idx d1) : data{ 1,d1,1 } {}
		shp3(idx d1, idx d2) : data{ 1,d1,d2 } {}
		shp3(idx d1, idx d2, idx d3) : data{ d1,d2,d3 } {}
		shp3(const dim1<3>& d) : data(d) {}
		const idx& operator[](idx i) const { return data[i]; }
		idx& operator[](idx i) { return data[i]; }
		operator auto() {
			return data;
		}
		dim1<3> data;
	};
	using shp4 = dim1<4>;
	inline constexpr idx c_dim(idx i, idx k, idx s, idx p) { return (i + 2 * p - k) / s + 1; }
	inline constexpr idx t_dim(idx i, idx k, idx s, idx p) { return (i - 1) * s + k - 2 * p; }
	inline constexpr idx c_pad(idx i, idx k, idx s, idx o) { return (i - 2 - o * s + s + k) / 2; }
	inline constexpr idx t_pad(idx i, idx k, idx s, idx o) { return ((i - 1) * s - o + k) / 2; }
	inline constexpr idx ti_pad(idx i, idx k, idx s, idx o) { return (k + s * (1 - i) + o - 2) / 2; }

	inline Tensor& random_r(Tensor& c, float min = 0.f, float max = 1.f) {
		c.setRandom();
		c = (max - min) * c + min;
		return c;
	}
	//multiply all matrix combinations stored as a0b0,a0b1,a1b0,a1b1....
	inline Tensor& mul_r(Tensor& c, const Tensor& a, const Tensor& b, shp2 dims = { 1, 0 }) {
		for (int i = 0; i < a.dimension(0); i++) {
			for (int j = 0; j < b.dimension(0); j++) {
				c.chip(i * b.dimension(0) + j, 0) = a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{ dims });
			}
		}
		return c;
	}
	inline Tensor& mul_r(Tensor& c, const Tensor& a, const Tensor& b, dim1<3> shape, shp2 dims) {
		auto y = c.reshape(shape);
		for (int i = 0; i < a.dimension(0); i++) {
			for (int j = 0; j < b.dimension(0); j++) {
				y.chip(i * b.dimension(0) + j, 0) = a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{ dims });
			}
		}
		return c;
	}
	//fma operation
	inline Tensor& fma_r(Tensor& d, const Tensor& a, const Tensor& b, const Tensor& c, shp2 dims = { 1, 0 }) {
		for (int i = 0; i < a.dimension(0); i++) {
			for (int j = 0; j < b.dimension(0); j++) {
				d.chip(i * b.dimension(0) + j, 0) = a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{ dims }) + c.chip(i * b.dimension(0) + j, 0);
			}
		}
		return d;
	}
	//multiply all matrix combinations and accumulate as a0b0 + a0b1, a1b0 + a1b1....
	inline Tensor& mul_acc_r(Tensor& c, const Tensor& a, const Tensor& b, shp2 dims = { 1, 0 }) {
		c.setZero();
		for (int i = 0; i < a.dimension(0); i++) {
			for (int j = 0; j < b.dimension(0); j++) {
				c.chip(i, 0) += a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{dims});
			}
		}
		return c;
	}

	inline Tensor& conv_r(Tensor& c, const Tensor& a, const Tensor& b, shp2 str, shp2 pad = { 0,0 }) {
		c.setZero();
		idx d0 = b.dimension(0) / a.dimension(0);
		dim1<2> st{str.first, str.second};
		dim2<2> pa{ shp2{ pad.first, pad.first}, shp2{ pad.second, pad.second }};
		for (int i = 0; i < d0; i++) {
			for (int j = 0; j < a.dimension(0); j++) {
				c.chip(i, 0) += a.chip(j, 0).pad(pa).convolve(b.chip(i * a.dimension(0) + j, 0), dim1<2>{0, 1}).stride(st);
			}
		}
		return c;
	}
	inline Tensor& conv_r(Tensor& c, const Tensor& a, const Tensor& b, dim1<3> shape, shp2 str, shp2 pad) {
		c.setZero();
		idx d0 = b.dimension(0) / a.dimension(0);
		dim1<2> st{str.first, str.second};
		dim2<2> pa{ shp2{ pad.first, pad.first}, shp2{ pad.second, pad.second }};
		for (int i = 0; i < d0; i++) { // 8 / 2 = 4
			for (int j = 0; j < a.dimension(0); j++) { //2							
				c.reshape(shape).chip(i, 0) += a.chip(j, 0).pad(pa).convolve(b.chip(i * a.dimension(0) + j, 0), dim1<2>{0, 1}).stride(st);
			}
		}
		return c;
	}

	inline Tensor iconv_r(Tensor& c, const Tensor& a, const Tensor& b, shp2 str = 1, shp2 pad = 0) {
		dim1<2> st{str.first, str.second};
		dim2<2> pa{ shp2{ pad.first, pad.first}, shp2{ pad.second, pad.second }};
		for (int i = 0; i < b.dimension(0); i++) { //4
			for (int j = 0; j < a.dimension(0); j++) { //2
				c.chip(i * a.dimension(0) + j, 0) = a.chip(j, 0).pad(pa).convolve(b.chip(i, 0), dim1<2>{0, 1}).stride(st);
			}
		}
		return c;
	}

	inline Tensor& pool_max_r(Tensor& c, const Tensor& a, shp2 ker, shp2 str = { 1,1 }) {
		idx d0 = a.dimension(0);
		idx d1 = c_dim(a.dimension(1), ker.first, str.first, 0);
		idx d2 = c_dim(a.dimension(2), ker.second, str.second, 0);
		dim1<2> st{str.first, str.second};
		dim1<2> ks{ker.first, ker.second};
		for (int i = 0; i < d0; i++) {
			for (int k = 0; k < d2; k++) {
				for (int j = 0; j < d1; j++) {
					dim1<2> off{j* st[0], k* st[1]};
					fsca y = a.chip(i, 0).slice(off, ks).maximum();
					c(i, j, k) = y(0);
				}
			}
		}
		return c;
	}

	inline Tensor& pool_avg_r(Tensor& c, const Tensor& a, shp2 ker, shp2 str = { 1,1 }) {
		idx d0 = a.dimension(0);
		idx d1 = c_dim(a.dimension(1), ker.first, str.first, 0);
		idx d2 = c_dim(a.dimension(2), ker.second, str.second, 0);
		dim1<2> st{str.first, str.second};
		dim1<2> ks{ker.first, ker.second};
		float mult = (1.f / (ks[0] * ks[1]));
		for (int i = 0; i < d0; i++) {
			for (int k = 0; k < d2; k++) {
				for (int j = 0; j < d1; j++) {
					dim1<2> off{j* st[0], k* st[1]};
					fsca y = a.chip(i, 0).slice(off, ks).sum();
					c(i, j, k) = y(0) * mult;
				}
			}
		}
		return c;
	}

	//multiply all matrix combinations stored as a0b0,a0b1,a1b0,a1b1....
	inline Tensor mul(const Tensor& a, const Tensor& b, shp2 dims = { 1, 0 }) {
		Tensor c(a.dimension(0) * b.dimension(0), dims.first ? a.dimension(1) : a.dimension(2), dims.second ? b.dimension(1) : b.dimension(2));
		return mul_r(c, a, b, dims);
	}
	//fma operation
	inline Tensor fma(const Tensor& a, const Tensor& b, const Tensor& c, shp2 dims = { 1, 0 }) {
		Tensor d(c.dimensions());
		return fma_r(d, a, b, c, dims);
	}
	//multiply all matrix combinations and accumulate as a0b0 + a0b1, a1b0 + a1b1....
	inline Tensor mul_acc(const Tensor& a, const Tensor& b, shp2 dims = { 1, 0 }) {
		Tensor c(a.dimension(0), dims.first ? a.dimension(1) : a.dimension(2), dims.second ? b.dimension(1) : b.dimension(2));
		return mul_acc_r(c, a, b, dims);
	}

	inline Tensor iconv(const Tensor& a, const Tensor& b, shp2 str = 1, shp2 pad = 0) {
		idx d0 = b.dimension(0) * a.dimension(0);
		idx d1 = c_dim(a.dimension(1), b.dimension(1), str.first, pad.first);
		idx d2 = c_dim(a.dimension(2), b.dimension(2), str.second, pad.second);
		Tensor c(d0, d1, d2);
		return iconv_r(c, a, b, str, pad);
	}

	inline Tensor conv(const Tensor& a, const Tensor& b, shp2 str = 1, shp2 pad = 0) {
		idx d0 = b.dimension(0) / a.dimension(0);
		idx d1 = c_dim(a.dimension(1), b.dimension(1), str.first, pad.first);
		idx d2 = c_dim(a.dimension(2), b.dimension(2), str.second, pad.second);
		Tensor c(d0, d1, d2);
		return conv_r(c, a, b, str, pad);
	}

	inline Tensor pool_max(const Tensor& a, shp2 ker = 2, shp2 str = 1) {
		idx d0 = a.dimension(0);
		idx d1 = c_dim(a.dimension(1), ker.first, str.first, 0);
		idx d2 = c_dim(a.dimension(2), ker.second, str.second, 0);
		Tensor c(d0, d1, d2);
		return pool_max_r(c, a, ker, str);
	}

	inline Tensor pool_avg(const Tensor& a, shp2 ker = 2, shp2 str = 1) {
		idx d0 = a.dimension(0);
		idx d1 = c_dim(a.dimension(1), ker.first, str.first, 0);
		idx d2 = c_dim(a.dimension(2), ker.second, str.second, 0);
		Tensor c(d0, d1, d2);
		return pool_avg_r(c, a, ker, str);
	}
	//convolve multiple input channels with multiple filters !!!filters have to be stored in the same order as channels eg - 3 channels, 9 filters = 3 outputs!!!


	inline void print_np(const auto& ten) {
		print("Tensor");
		for (int i = 0; i < ten.NumDimensions; i++) {
			print(ten.dimension(i));
		}
		print("\n");
		println(ten.format(Eigen::TensorIOFormat::Numpy()));
		print("\n-------------------------------\n");
	}
}



