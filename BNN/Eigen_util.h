#pragma once
#include "Misc.h"
#include <unsupported/Eigen/CXX11/Tensor>
namespace BNN {
	using Tensor = Eigen::Tensor<float, 3>;
	using Tenarr = Eigen::Tensor<float, 4>;
	using fsca = Eigen::TensorFixedSize<float, Eigen::Sizes<>>;
	using idx = Eigen::Index;
	using pair = Eigen::IndexPair<idx>;
	template <size_t N>
	using dim1 = Eigen::array<idx, N>;
	template <size_t N>
	using dim2 = Eigen::array<pair, N>;
	template <class T, size_t N>
	using dimx = Eigen::array<T, N>;
	
	inline constexpr idx c_dim(idx is, idx ks, idx st, idx pa) { return (is + 2 * pa - ks) / st + 1; }
	inline constexpr idx c_pad(idx os, idx is, idx ks, idx st) { return ((os - 1) - is + ks) / 2; }
	inline constexpr idx c_rem(idx is, idx ks, idx st, idx pa) { return (is + 2 * pa - ks) % st; }

	inline Tensor& random_e(Tensor& c, float min = 0.f, float max = 1.f) {
		c.setRandom();
		c = (max - min) * c + min;
		return c;
	}
	//multiply all matrix combinations stored as a0b0,a0b1,a1b0,a1b1....
	inline Tensor& mul_e(Tensor& c, const Tensor& a, const Tensor& b, pair dims = { 1, 0 }) {
		//c.resize(a.dimension(0) * b.dimension(0), dims.first ? a.dimension(1) : a.dimension(2), dims.second ? b.dimension(1) : b.dimension(2));
		for (int i = 0; i < a.dimension(0); i++) {
			for (int j = 0; j < b.dimension(0); j++) {
				c.chip(i * b.dimension(0) + j, 0) = a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{ dims });
			}
		}
		return c;
	}
	//fma operation
	inline Tensor& fma_e(Tensor& c, const Tensor& a, const Tensor& b, const Tensor& d, pair dims = { 1, 0 }) {
		//c.resize(a.dimension(0) * b.dimension(0), dims.first ? a.dimension(1) : a.dimension(2), dims.second ? b.dimension(1) : b.dimension(2));
		for (int i = 0; i < a.dimension(0); i++) {
			for (int j = 0; j < b.dimension(0); j++) {
				c.chip(i * b.dimension(0) + j, 0) = a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{ dims }) + d.chip(i * b.dimension(0) + j, 0);
			}
		}
		return c;
	}
	//multiply all matrix combinations and accumulate as a0b0 + a0b1, a1b0 + a1b1....
	inline Tensor& mul_acc_e(Tensor& c, const Tensor& a, const Tensor& b, pair dims = { 1, 0 }) {
		//c.resize(a.dimension(0), dims.first ? a.dimension(1) : a.dimension(2), dims.second ? b.dimension(1) : b.dimension(2));
		for (int i = 0; i < a.dimension(0); i++) {
			c.chip(i, 0).setZero();
			for (int j = 0; j < b.dimension(0); j++) {
				c.chip(i, 0) += a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{dims});
			}
		}
		return c;
	}

	inline Tensor& conv_e(Tensor& c, const Tensor& a, const Tensor& b, pair str, pair pad = { 0,0 }) {
		idx d0 = b.dimension(0) / a.dimension(0);
		idx d1 = c_dim(a.dimension(1), b.dimension(1), str.first, pad.first);
		idx d2 = c_dim(a.dimension(2), b.dimension(2), str.second, pad.second);
		//c.resize(d0, d1, d2);
		dim1<2> st{str.first, str.second};
		dim2<2> pa{ pair{ pad.first, pad.first}, pair{ pad.second, pad.second }};
		for (int i = 0; i < d0; i++) {
			c.chip(i, 0).setZero();
			for (int j = 0; j < a.dimension(0); j++) {
				c.chip(i, 0) += a.chip(j, 0).pad(pa).convolve(b.chip(i * a.dimension(0) + j, 0), dim1<2>{0, 1}).stride(st);
			}
		}
		return c;
	}

	inline Tensor& pool_max_e(Tensor& c, const Tensor& a, pair ker, pair str = { 1,1 }) {
		idx d0 = a.dimension(0);
		idx d1 = c_dim(a.dimension(1), ker.first, str.first, 0);
		idx d2 = c_dim(a.dimension(2), ker.second, str.second, 0);
		//c.resize(d0, d1, d2);
		dim1<2> st{str.first, str.second};
		dim1<2> ks{ker.first, ker.second};
		for (int i = 0; i < d0; i++) {
			auto x = a.chip(i, 0);
			for (int k = 0; k < d2; k++) {
				for (int j = 0; j < d1; j++) {
					dim1<2> off{j* st[0], k* st[1]};
					fsca y = x.slice(off, ks).maximum();
					c(i, j, k) = y(0);
				}
			}
		}
		return c;
	}

	inline Tensor& pool_avg_e(Tensor& c, const Tensor& a, pair ker, pair str = { 1,1 }) {
		idx d0 = a.dimension(0);
		idx d1 = c_dim(a.dimension(1), ker.first, str.first, 0);
		idx d2 = c_dim(a.dimension(2), ker.second, str.second, 0);
		////c.resize(d0, d1, d2);
		dim1<2> st{str.first, str.second};
		dim1<2> ks{ker.first, ker.second};
		float mult = (1.f / (ks[0] * ks[1]));
		for (int i = 0; i < d0; i++) {
			auto x = a.chip(i, 0);
			for (int k = 0; k < d2; k++) {
				for (int j = 0; j < d1; j++) {
					dim1<2> off{j* st[0], k* st[1]};
					fsca y = x.slice(off, ks).sum();
					c(i, j, k) = y(0) * mult;
				}
			}
		}
		return c;
	}
	//convolve multiple input channels with multiple filters !!!filters have to be stored in the same order as channels eg - 3 channels, 9 filters = 3 outputs!!!
	inline Tensor& conv_e(Tensor& c, const Tensor& a, const Tensor& b, idx str = 1, idx pad = 0) {
		return conv_e(c, a, b, { str,str }, { pad,pad });
	}
	inline Tensor& pool_max_e(Tensor& c, const Tensor& a, idx ker = 2, idx str = 1) {
		return pool_max_e(c, a, { ker,ker }, { str,str });
	}
	inline Tensor& pool_avg_e(Tensor& c, const Tensor& a, idx ker = 2, idx str = 1) {
		return pool_avg_e(c, a, { ker,ker }, { str,str });
	}




	//multiply all matrix combinations stored as a0b0,a0b1,a1b0,a1b1....
	inline Tensor mul(const Tensor& a, const Tensor& b, pair dims = { 1, 0 }) {
		Tensor c(a.dimension(0) * b.dimension(0), dims.first ? a.dimension(1) : a.dimension(2), dims.second ? b.dimension(1) : b.dimension(2));
		for (int i = 0; i < a.dimension(0); i++) {
			for (int j = 0; j < b.dimension(0); j++) {
				c.chip(i * b.dimension(0) + j, 0) = a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{ dims });
			}
		}
		return c;
	}
	//fma operation
	inline Tensor fma(const Tensor& a, const Tensor& b, const Tensor& c, pair dims = { 1, 0 }) {
		Tensor d(a.dimension(0) * b.dimension(0), dims.first ? a.dimension(1) : a.dimension(2), dims.second ? b.dimension(1) : b.dimension(2));
		for (int i = 0; i < a.dimension(0); i++) {
			for (int j = 0; j < b.dimension(0); j++) {
				d.chip(i * b.dimension(0) + j, 0) = a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{ dims }) + c.chip(i * b.dimension(0) + j, 0);
			}
		}
		return d;
	}
	//multiply all matrix combinations and accumulate as a0b0 + a0b1, a1b0 + a1b1....
	inline Tensor mul_acc(const Tensor& a, const Tensor& b, pair dims = { 1, 0 }) {
		Tensor c(a.dimension(0), dims.first ? a.dimension(1) : a.dimension(2), dims.second ? b.dimension(1) : b.dimension(2));
		for (int i = 0; i < a.dimension(0); i++) {
			c.chip(i, 0).setZero();
			for (int j = 0; j < b.dimension(0); j++) {
				c.chip(i, 0) += a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{dims});
			}
		}
		return c;
	}

	inline Tensor iconv(const Tensor& a, const Tensor& b, pair str, pair pad = { 0,0 }) {
		idx d0 = b.dimension(0) * a.dimension(0);
		idx d1 = c_dim(a.dimension(1), b.dimension(1), str.first, pad.first);
		idx d2 = c_dim(a.dimension(2), b.dimension(2), str.second, pad.second);
		Tensor c(d0, d1, d2);
		dim1<2> st{str.first, str.second};
		dim2<2> pa{ pair{ pad.first, pad.first}, pair{ pad.second, pad.second }};
		for (int i = 0; i < b.dimension(0); i++) {
			auto x = b.chip(i, 0);
			for (int j = 0; j < a.dimension(0); j++) {
				c.chip(i * a.dimension(0) + j, 0) = a.chip(j, 0).pad(pa).convolve(x, dim1<2>{0, 1}).stride(st);
			}
		}
		return c;
	}

	inline Tensor conv(const Tensor& a, const Tensor& b, pair str, pair pad = { 0,0 }) {
		idx d0 = b.dimension(0) / a.dimension(0);
		idx d1 = c_dim(a.dimension(1), b.dimension(1), str.first, pad.first);
		idx d2 = c_dim(a.dimension(2), b.dimension(2), str.second, pad.second);
		Tensor c(d0, d1, d2);
		dim1<2> st{str.first, str.second};
		dim2<2> pa{ pair{ pad.first, pad.first}, pair{ pad.second, pad.second }};
		for (int i = 0; i < d0; i++) {
			c.chip(i, 0).setZero();
			for (int j = 0; j < a.dimension(0); j++) {
				c.chip(i, 0) += a.chip(j, 0).pad(pa).convolve(b.chip(i * a.dimension(0) + j, 0), dim1<2>{0, 1}).stride(st);
			}
		}
		return c;
	}
	
	inline Tensor pool_max(const Tensor& a, pair ker, pair str = { 1,1 }) {
		idx d0 = a.dimension(0);
		idx d1 = c_dim(a.dimension(1), ker.first, str.first, 0);
		idx d2 = c_dim(a.dimension(2), ker.second, str.second, 0);
		Tensor c(d0, d1, d2);
		dim1<2> st{str.first, str.second};
		dim1<2> ks{ker.first, ker.second};
		for (int i = 0; i < d0; i++) {
			auto x = a.chip(i, 0);
			for (int k = 0; k < d2; k++) {
				for (int j = 0; j < d1; j++) {
					dim1<2> off{j* st[0], k* st[1]};
					fsca y = x.slice(off, ks).maximum();
					c(i, j, k) = y(0);
				}
			}
		}
		return c;
	}

	inline Tensor pool_avg(const Tensor& a, pair ker, pair str = { 1,1 }) {
		idx d0 = a.dimension(0);
		idx d1 = c_dim(a.dimension(1), ker.first, str.first, 0);
		idx d2 = c_dim(a.dimension(2), ker.second, str.second, 0);
		Tensor c(d0, d1, d2);
		dim1<2> st{str.first, str.second};
		dim1<2> ks{ker.first, ker.second};
		float mult = (1.f / (ks[0] * ks[1]));
		for (int i = 0; i < d0; i++) {
			auto x = a.chip(i, 0);
			for (int k = 0; k < d2; k++) {
				for (int j = 0; j < d1; j++) {
					dim1<2> off{j* st[0], k* st[1]};
					fsca y = x.slice(off, ks).sum();
					c(i, j, k) = y(0) * mult;
				}
			}
		}
		return c;
	}
	//convolve multiple input channels with multiple filters !!!filters have to be stored in the same order as channels eg - 3 channels, 9 filters = 3 outputs!!!
	inline Tensor conv(const Tensor& a, const Tensor& b, idx str = 1, idx pad = 0) {
		return conv(a, b, { str,str }, { pad,pad });
	}
	inline Tensor pool_max(const Tensor& a, idx ker = 2, idx str = 1) {
		return pool_max(a, { ker,ker }, { str,str });
	}
	inline Tensor pool_avg(const Tensor& a, idx ker = 2, idx str = 1) {
		return pool_avg(a, { ker,ker }, { str,str });
	}


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



