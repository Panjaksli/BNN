#pragma once
#include "Eigen_util.h"
namespace BNN {

	void conv2d(Tensor& out, const Tensor& inp, const Tensor& ker, shp2 st, shp2 pa, shp2 dil = 1);
	void conv2d_igrad(Tensor& inp, const Tensor& out, const Tensor& ker, shp2 st, shp2 pa, shp2 dil = 1);
	void conv2d_wgrad(Tensor& ker, const Tensor& inp, const Tensor& out, shp2 st, shp2 pa, shp2 dil = 1);

	void convolve(Tensor& c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa);
	//fixing a fuckup where I was convolving incorrect filters in backprop, also reverse operation is included inhouse
	//stupid me
	//It was convolving for example 4 output filters by 4 first filters of the kernel which werent the filters corresponding to the input !!!
	void rev_convolve(Tensor& c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa);
	//Convolve all filter combinations, N channels, K filters, N*K output channels
	
	void all_convolve(Tensor& c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa);
	//Convolve all combinations and accumulate to output
	
	void acc_convolve(Tensor& c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa);
	//Convolve each input with each channel
	
	void convolve_1to1(Tensor& c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa);
	
	void rev_convolve_1to1(Tensor& c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa);
	
	void acc_convolve_1to1(Tensor& c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa);
	//bilinear resize
	void resize_r(TensRef y, const Tensor& x, Interpol filter);
	//multiply all matrix combinations stored as a0b0,a0b1,a1b0,a1b1....
	
	inline void mul_r(Tensor& c, const Tensor& a, const Tensor& b, shp2 dims = { 1, 0 }) {
		
		for(idx i = 0; i < a.dimension(2); i++) {
			for(idx j = 0; j < b.dimension(2); j++) {
				c.chip(i * b.dimension(2) + j, 2) = a.chip(i, 2).contract(b.chip(j, 2), dim2<1>{ dims });
			}
		}
	}
	//fma operation
	
	inline void fma_r(Tensor& c, const Tensor& a, const Tensor& b, const Tensor& d, shp2 dims = { 1, 0 }) {
		for(idx i = 0; i < a.dimension(2); i++) {
			for(idx j = 0; j < b.dimension(2); j++) {
				c.chip(i * b.dimension(2) + j, 2) = a.chip(i, 2).contract(b.chip(j, 2), dim2<1>{ dims }) + d.chip(i * b.dimension(2) + j, 2);
			}
		}
		//return d;
	}
	
	inline void acc_mul(Tensor& c, const Tensor& a, const Tensor& b, shp2 dims = { 1, 0 }) {
		
		for(idx i = 0; i < a.dimension(2); i++) {
			for(idx j = 0; j < b.dimension(2); j++) {
				c.chip(i * b.dimension(2) + j, 2) += a.chip(i, 2).contract(b.chip(j, 2), dim2<1>{ dims });
			}
		}
		//return d;
	}
	//multiply all matrix combinations and accumulate as a0b0 + a0b1, a1b0 + a1b1....
	
	inline void mul_acc_r(Tensor& c, const Tensor& a, const Tensor& b, shp2 dims = { 1, 0 }) {
		
		c.setZero();
		for(idx i = 0; i < a.dimension(2); i++) {
			for(idx j = 0; j < b.dimension(2); j++) {
				c.chip(i, 2) += a.chip(i, 2).contract(b.chip(j, 2), dim2<1>{dims});
			}
		}
	}
	//convolute and accumulate filters -> b / a filters (b HAS to be multiple of a)
	inline void pool_max_r(Tensor& c, const Tensor& a, shp2 ker, shp2 str = 1) {
		idx d0 = a.dimension(0);
		idx d1 = c_dim(a.dimension(1), ker[0], str[0], 0);
		idx d2 = c_dim(a.dimension(2), ker[1], str[1], 0);
		dim1<2> st{ str[0], str[1] };
		dim1<2> ks{ ker[0], ker[1] };
		for(idx k = 0; k < d2; k++) {
			for(idx j = 0; j < d1; j++) {
				for(idx i = 0; i < d0; i++) {
					dim1<2> off{ j * st[0], k * st[1] };
					c(i, j, k) = fsca(a.chip(i, 0).slice(off, ks).maximum()).coeff();
				}
			}
		}
	}
	//multiply all matrix combinations stored as a0b0,a0b1,a1b0,a1b1....
	inline Tensor mul(const Tensor& a, const Tensor& b, shp2 dims = { 1, 0 }) {
		Tensor c(a.dimension(0) * b.dimension(0), dims[0] ? a.dimension(1) : a.dimension(2), dims[1] ? b.dimension(1) : b.dimension(2));
		mul_r(c, a, b, dims);
		return c;
	}
	//fma operation
	inline Tensor fma(const Tensor& a, const Tensor& b, const Tensor& c, shp2 dims = { 1, 0 }) {
		Tensor d(c.dimensions());
		fma_r(d, a, b, c, dims);
		return d;
	}
	//multiply all matrix combinations and accumulate as a0b0 + a0b1, a1b0 + a1b1....
	inline Tensor mul_acc(const Tensor& a, const Tensor& b, shp2 dims = { 1, 0 }) {
		Tensor c(a.dimension(0), dims[0] ? a.dimension(1) : a.dimension(2), dims[1] ? b.dimension(1) : b.dimension(2));
		mul_acc_r(c, a, b, dims);
		return c;
	}

	inline Tensor aconv(const Tensor& a, const Tensor& b, shp2 str = 1, shp2 pad = 0) {
		idx d0 = b.dimension(0) * a.dimension(0);
		idx d1 = c_dim(a.dimension(1), b.dimension(1), str[0], pad[0]);
		idx d2 = c_dim(a.dimension(2), b.dimension(2), str[1], pad[1]);
		Tensor c(d0, d1, d2);
		all_convolve(c, a, b, str, pad);
		return c;
	}

	inline Tensor conv(const Tensor& a, const Tensor& b, shp2 str = 1, shp2 pad = 0) {
		idx d0 = b.dimension(0) / a.dimension(0);
		idx d1 = c_dim(a.dimension(1), b.dimension(1), str[0], pad[0]);
		idx d2 = c_dim(a.dimension(2), b.dimension(2), str[1], pad[1]);
		Tensor c(d0, d1, d2);
		convolve(c, a, b, str, pad);
		return c;
	}
	inline Tensor conv_1to1(const Tensor& a, const Tensor& b, shp2 str = 1, shp2 pad = 0) {
		idx d0 = a.dimension(0);
		idx d1 = c_dim(a.dimension(1), b.dimension(1), str[0], pad[0]);
		idx d2 = c_dim(a.dimension(2), b.dimension(2), str[1], pad[1]);
		Tensor c(d0, d1, d2);
		convolve_1to1(c, a, b, str, pad);
		return c;
	}
	inline Tensor pool_max(const Tensor& a, shp2 ker = 2, shp2 str = 1) {
		idx d0 = a.dimension(0);
		idx d1 = c_dim(a.dimension(1), ker[0], str[0], 0);
		idx d2 = c_dim(a.dimension(2), ker[1], str[1], 0);
		Tensor c(d0, d1, d2);
		pool_max_r(c, a, ker, str);
		return c;
	}
	inline Tensor resize(const Tensor& x, double s1, double s2, Interpol filter = Cubic) {
		Tensor y(x.dimension(0), idx(x.dimension(1) * s1), idx(x.dimension(2) * s2));
		resize_r(y, x, filter);
		return y;
	}
}