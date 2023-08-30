#pragma once
#include "Eigen_util.h"
namespace BNN {
	template <int cache = 4096>
	inline void convolve(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
		idx ich = a.dimension(0);
		idx och = b.dimension(0) / ich;
		if(ich <= 1 && och <= 1) {
			if(st[0] <= 1 && st[1] <= 1) {
				dim2<2> pad{ shp2{ pa[0], pa[0]}, shp2{ pa[1], pa[1] } };
				c.data.reshape(c.dim).chip(0, 0) = a.chip(0, 0).pad(pad).convolve(b.chip(0, 0), dim1<2>{0, 1});
			}
			else {
				float tmp;
				for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
					for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
						tmp = 0;
						idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
						idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
						for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
							for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
								tmp += a(0, j + m, i + l) * b(0, m, l);
							}
						}
						c(0, p, o) = tmp;
					}
				}
			}
		}
		else if(ich <= 1) {
			float tmp[cache];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, 4 * och);
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx k = 0; k < och; k++) {
								tmp[k] += a(0, j + m, i + l) * b(k, m, l);
							}
						}
					}
					memmove(&c(0, p, o), tmp, och * 4);
				}
			}
		}
		else if(och <= 1) {
			float tmp = 0;
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					tmp = 0;
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx n = 0; n < ich; n++) {
								tmp += a(n, j + m, i + l) * b(n, m, l);
							}
						}
					}
					c(0, p, o) = tmp;
				}
			}
		}
		else {
			float tmp[cache];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, 4 * och);
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx k = 0; k < och; k++) {
								for(idx n = 0; n < ich; n++) {
									tmp[k] += a(n, j + m, i + l) * b(k * ich + n, m, l);
								}
							}
						}
					}
					memmove(&c(0, p, o), tmp, och * 4);
				}
			}
		}
	}
	//fixing a fuckup where I was convolving incorrect filters in backprop, also reverse operation is included inhouse
	//stupid me
	//It was convolving for example 4 output filters by 4 first filters of the kernel which werent the filters corresponding to the input !!!
	template <int cache = 4096>
	inline void rev_convolve(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
		idx ich = a.dimension(0);
		idx och = b.dimension(0) / ich;
		idx bdm = b.dimension(1);
		idx bdl = b.dimension(2);
		if(ich <= 1 && och <= 1) {
			if(st[0] <= 1 && st[1] <= 1) {
				dim2<2> pad{ shp2{ pa[0], pa[0]}, shp2{ pa[1], pa[1] } };
				c.data.reshape(c.dim).chip(0, 0) = a.chip(0, 0).pad(pad).convolve(b.chip(0, 0).reverse(dimx<bool, 2>{true, true}), dim1<2>{0, 1});
			}
			else {
				float tmp;
				for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - bdl + 1); i += st[1], o++) {
					for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - bdm + 1); j += st[0], p++) {
						tmp = 0;
						idx clip_l = max(0, i + bdl - a.dimension(2));
						idx clip_m = max(0, j + bdm - a.dimension(1));
						for(idx l = max(-i, 0); l < bdl - clip_l; l++) {
							for(idx m = max(-j, 0); m < bdm - clip_m; m++) {
								tmp += a(0, j + m, i + l) * b(0, bdm - 1 - m, bdl - 1 - l);
							}
						}
						c(0, p, o) = tmp;
					}
				}
			}
		}
		else if(ich <= 1) {
			float tmp[cache];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - bdl + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - bdm + 1); j += st[0], p++) {
					memset(tmp, 0, 4 * och);
					idx clip_l = max(0, i + bdl - a.dimension(2));
					idx clip_m = max(0, j + bdm - a.dimension(1));
					for(idx l = max(-i, 0); l < bdl - clip_l; l++) {
						for(idx m = max(-j, 0); m < bdm - clip_m; m++) {
							for(idx k = 0; k < och; k++) {
								tmp[k] += a(0, j + m, i + l) * b(k, bdm - 1 - m, bdl - 1 - l);
							}
						}
					}
					memmove(&c(0, p, o), tmp, och * 4);
				}
			}
		}
		else if(och <= 1) {
			float tmp = 0;
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - bdl + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - bdm + 1); j += st[0], p++) {
					tmp = 0;
					idx clip_l = max(0, i + bdl - a.dimension(2));
					idx clip_m = max(0, j + bdm - a.dimension(1));
					for(idx l = max(-i, 0); l < bdl - clip_l; l++) {
						for(idx m = max(-j, 0); m < bdm - clip_m; m++) {
							for(idx n = 0; n < ich; n++) {
								tmp += a(n, j + m, i + l) * b(n, bdm - 1 - m, bdl - 1 - l);
							}
						}
					}
					c(0, p, o) = tmp;
				}
			}
		}
		else {
			float tmp[cache];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - bdl + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - bdm + 1); j += st[0], p++) {
					memset(tmp, 0, 4 * och);
					idx clip_l = max(0, i + bdl - a.dimension(2));
					idx clip_m = max(0, j + bdm - a.dimension(1));
					for(idx l = max(-i, 0); l < bdl - clip_l; l++) {
						for(idx m = max(-j, 0); m < bdm - clip_m; m++) {
							for(idx n = 0; n < ich; n++) {
								for(idx k = 0; k < och; k++) {
									tmp[k] += a(n, j + m, i + l) * b(n * och + k, bdm - 1 - m, bdl - 1 - l);
								}
							}
						}
					}
					memmove(&c(0, p, o), tmp, och * 4);
				}
			}
		}
	}

	template <int cache = 4>
	inline void all_convolve(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
		idx ach = a.dimension(0);
		idx bch = b.dimension(0);
		idx och = ach * bch;
		if(ach <= 1 && bch <= 1) {
			if(st[0] <= 1 && st[1] <= 1) {
				dim2<2> pad{ shp2{ pa[0], pa[0]}, shp2{ pa[1], pa[1] } };
				c.data.reshape(c.dim).chip(0, 0) = a.chip(0, 0).pad(pad).convolve(b.chip(0, 0), dim1<2>{0, 1});
			}
			else {
				float tmp;
				for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
					for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
						tmp = 0;
						idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
						idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
						for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
							for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
								tmp += a(0, j + m, i + l) * b(0, m, l);
							}
						}
						c(0, p, o) = tmp;
					}
				}
			}
		}
		else if(ach <= 1) {
			float tmp[cache];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, 4 * och);
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx k = 0; k < bch; k++) {
								tmp[k] += a(0, j + m, i + l) * b(k, m, l);
							}
						}
					}
					memmove(&c(0, p, o), tmp, och * 4);
				}
			}
		}
		else if(bch <= 1) {
			float tmp[cache];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, 4 * och);
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx n = 0; n < ach; n++) {
								tmp[n] += a(n, j + m, i + l) * b(0, m, l);
							}
						}
					}
					memmove(&c(0, p, o), tmp, och * 4);
				}
			}
		}
		else {
			float tmp[cache];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, 4 * och);
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx k = 0; k < bch; k++) {
								for(idx n = 0; n < ach; n++) {
									tmp[k * ach + n] += a(n, j + m, i + l) * b(k, m, l);
								}
							}
						}
					}
					memmove(&c(0, p, o), tmp, och * 4);
				}
			}
		}
	}
	template <int cache = 4096>
	inline void acc_convolve(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
		idx ach = a.dimension(0);
		idx bch = b.dimension(0);
		idx och = ach * bch;
		if(ach <= 1 && bch <= 1) {
			if(st[0] <= 1 && st[1] <= 1) {
				dim2<2> pad{ shp2{ pa[0], pa[0]}, shp2{ pa[1], pa[1] } };
				c.data.reshape(c.dim).chip(0, 0) += a.chip(0, 0).pad(pad).convolve(b.chip(0, 0), dim1<2>{0, 1});
			}
			else {
				float tmp;
				for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
					for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
						tmp = 0;
						idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
						idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
						for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
							for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
								tmp += a(0, j + m, i + l) * b(0, m, l);
							}
						}
						c(0, p, o) += tmp;
					}
				}
			}
		}
		else if(ach <= 1) {
			float tmp[cache];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, 4 * och);
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx k = 0; k < bch; k++) {
								tmp[k] += a(0, j + m, i + l) * b(k, m, l);
							}
						}
					}
					for(idx ch = 0; ch < och; ch++)
						c(ch, p, o) += tmp[ch];
				}
			}
		}
		else if(bch <= 1) {
			float tmp[cache];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, 4 * och);
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx n = 0; n < ach; n++) {
								tmp[n] += a(n, j + m, i + l) * b(0, m, l);
							}
						}
					}
					for(idx ch = 0; ch < och; ch++)
						c(ch, p, o) += tmp[ch];
				}
			}
		}
		else {
			float tmp[cache];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, 4 * och);
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx k = 0; k < bch; k++) {
								for(idx n = 0; n < ach; n++) {
									tmp[k * ach + n] += a(n, j + m, i + l) * b(k, m, l);
								}
							}
						}
					}
					for(idx ch = 0; ch < och; ch++)
						c(ch, p, o) += tmp[ch];
				}
			}
		}
	}
	//bilinear resize
	template<int cache = 1024>
	void resize_r(Reshape y, const Tensor& x, Interpol filter) {
		float tmp[cache];
		float s1 = y.dim[1] > 0 ? float(x.dimension(1)) / (y.dim[1]) : 0;
		float s2 = y.dim[2] > 0 ? float(x.dimension(2)) / (y.dim[2]) : 0;
		if(filter == Nearest) {
			for(idx i = 0; i < y.dim[2]; i++) {
				idx li = (i + 0.5f) * s2;
				for(idx j = 0; j < y.dim[1]; j++) {
					idx lj = (j + 0.5f) * s1;
					for(idx k = 0; k < y.dim[0]; k++) {
						tmp[k] = x(k, lj, li);
					}
					memmove(&y(0, j, i), tmp, y.dim[0] * sizeof(float));
				}
			}
		}
		else if(filter == Linear) {
			for(idx i = 0; i < y.dim[2]; i++) {
				float fi = fmaxf((i + 0.5f) * s2 - 0.5f, 0);
				idx li = fi;
				idx hi = min(li + 1, x.dimension(2) - 1);
				float wi = fi - li;
				for(idx j = 0; j < y.dim[1]; j++) {
					float fj = fmaxf((j + 0.5f) * s1 - 0.5f, 0);
					idx lj = fj;
					idx hj = min(lj + 1, x.dimension(1) - 1);
					float wj = fj - lj;
					for(idx k = 0; k < y.dim[0]; k++) {
						float a = x(k, lj, li);
						float b = x(k, hj, li);
						float c = x(k, lj, hi);
						float d = x(k, hj, hi);
						tmp[k] = lerp(lerp(a, b, wj), lerp(c, d, wj), wi);
					}
					memmove(&y(0, j, i), tmp, y.dim[0] * sizeof(float));
				}
			}
		}
		else if(filter == Cubic) {
			for(idx i = 0; i < y.dim[2]; i++) {
				float yc = fmaxf((i + 0.5f) * s2 - 0.5f, 0);
				int y0 = yc;
				float yd = yc - y0;
				for(idx j = 0; j < y.dim[1]; j++) {
					float xc = fmaxf((j + 0.5f) * s1 - 0.5f, 0);
					int x0 = xc;
					float xd = xc - x0;
					for(idx k = 0; k < y.dim[0]; k++) {
						float p[16];
						for(int ii = 0; ii < 4; ii++) {
							for(int jj = 0; jj < 4; jj++) {
								p[ii * 4 + jj] = x(k, clamp(x0 + jj - 1, 0, x.dimension(1) - 1), clamp(y0 + ii - 1, 0, x.dimension(2) - 1));
							}
						}
						tmp[k] = clamp(bicerp(p, xd, yd), 0.f, 1.f);
					}
					memmove(&y(0, j, i), tmp, y.dim[0] * sizeof(float));
				}
			}
		}
	}
	//multiply all matrix combinations stored as a0b0,a0b1,a1b0,a1b1....
	template <class derived>
	inline void mul_r(const TensorBase<derived>& res, const Tensor& a, const Tensor& b, shp2 dims = { 1, 0 }) {
		auto& c = const_cast<Eigen::TensorBase<derived>&>(res);
		for(idx i = 0; i < a.dimension(0); i++) {
			for(idx j = 0; j < b.dimension(0); j++) {
				c.chip(i * b.dimension(0) + j, 0) = a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{ dims });
			}
		}
	}
	//fma operation
	template <class derived>
	inline void fma_r(const TensorBase<derived>& res, const Tensor& a, const Tensor& b, const Tensor& c, shp2 dims = { 1, 0 }) {
		auto& d = const_cast<Eigen::TensorBase<derived>&>(res);
		for(idx i = 0; i < a.dimension(0); i++) {
			for(idx j = 0; j < b.dimension(0); j++) {
				d.chip(i * b.dimension(0) + j, 0) = a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{ dims }) + c.chip(i * b.dimension(0) + j, 0);
			}
		}
		//return d;
	}
	template <class derived>
	inline void acc_mul(const TensorBase<derived>& res, const Tensor& a, const Tensor& b, shp2 dims = { 1, 0 }) {
		auto& c = const_cast<Eigen::TensorBase<derived>&>(res);
		for(idx i = 0; i < a.dimension(0); i++) {
			for(idx j = 0; j < b.dimension(0); j++) {
				c.chip(i * b.dimension(0) + j, 0) += a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{ dims });
			}
		}
		//return d;
	}
	//multiply all matrix combinations and accumulate as a0b0 + a0b1, a1b0 + a1b1....
	template <class derived>
	inline void mul_acc_r(const TensorBase<derived>& res, const Tensor& a, const Tensor& b, shp2 dims = { 1, 0 }) {
		auto& c = const_cast<Eigen::TensorBase<derived>&>(res);
		c.setZero();
		for(idx i = 0; i < a.dimension(0); i++) {
			for(idx j = 0; j < b.dimension(0); j++) {
				c.chip(i, 0) += a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{dims});
			}
		}
	}
	//convolute and accumulate filters -> b / a filters (b HAS to be multiple of a)

	inline void pool_max_r(Reshape c, const Tensor& a, shp2 ker, shp2 str = 1) {
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

	inline Tensor pool_max(const Tensor& a, shp2 ker = 2, shp2 str = 1) {
		idx d0 = a.dimension(0);
		idx d1 = c_dim(a.dimension(1), ker[0], str[0], 0);
		idx d2 = c_dim(a.dimension(2), ker[1], str[1], 0);
		Tensor c(d0, d1, d2);
		pool_max_r(c, a, ker, str);
		return c;
	}
	inline Tensor resize(const Tensor& x, double s1, double s2, Interpol filter = Cubic){
		Tensor y(x.dimension(0), idx(x.dimension(1) * s1), idx(x.dimension(2) * s2));
		resize_r(y, x, filter);
		return y;
	}
}