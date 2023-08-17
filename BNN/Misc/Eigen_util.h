#pragma once
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "Misc.h"

namespace BNN {
	using idx = int_fast32_t;
	using Eigen::TensorBase;
	using Eigen::TensorRef;

	using Tensor = Eigen::Tensor<float, 3, 0, idx>;
	using Tenarr = Eigen::Tensor<float, 4, 0, idx>;
	using fsca = Eigen::TensorFixedSize<float, Eigen::Sizes<>, 0, idx>;

	template <size_t N>
	using dim1 = Eigen::DSizes<idx, N>;
	//using shp2 = Eigen::IndexPair<idx>;
	struct shp1 {
		shp1() {}
		shp1(idx first) : first(first) {}
		const idx& operator[](idx i) const { return i ? first : first; }
		idx& operator[](idx i) { return i ? first : first; }
		operator dim1<3>() {
			return dim1<3>{1, first, 1};
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
			return dim1<3>{1, first, second};
		}
		friend shp2 operator-(shp2 x) { return shp2{ -x[0],-x[1] }; }
		friend shp2 operator+(shp2 x, shp2 y) { return shp2{ x[0] + y[0],x[1] + y[1] }; }
		friend shp2 operator-(shp2 x, shp2 y) { return shp2{ x[0] - y[0],x[1] - y[1] }; }
		idx first;
		idx second;
	};
	template <size_t N>
	using dim2 = Eigen::array<shp2, N>;
	template <class T, size_t N>
	using dimx = Eigen::array<T, N>;
	struct shp3 {
		shp3() {}
		shp3(idx d1) : elem{ 1,d1,1 } {}
		shp3(idx d1, idx d2) : elem{ 1,d1,d2 } {}
		shp3(idx d1, idx d2, idx d3) : elem{ d1,d2,d3 } {}
		shp3(const dim1<3>& d) : elem(d) {}
		const idx& operator[](idx i) const { return elem[i]; }
		idx& operator[](idx i) { return elem[i]; }
		operator auto() {
			return elem;
		}
		dim1<3> elem;
	};
	inline idx product(const dim1<3>& x) { return x[0] * x[1] * x[2]; }
	using shp4 = dim1<4>;
	inline constexpr idx c_dim(idx i, idx k, idx s, idx p) { return (i + 2 * p - k) / s + 1; }
	inline constexpr idx t_dim(idx i, idx k, idx s, idx p) { return (i - 1) * s + k - 2 * p; }

	inline void random_r(Tensor& c, float min = 0.f, float max = 1.f) {
		c.setRandom();
		c = (max - min) * c + min;
	}

	struct Reshape {
		Reshape(Tensor& data) : data(data), dim(data.dimensions()) {}
		Reshape(Tensor& data, shp3 dim) : data(data), dim(dim) {}
		const float& operator() (idx i, idx j, idx k)const { return data.data()[i + j * dim[0] + k * dim[0] * dim[1]]; }
		float& operator() (idx i, idx j, idx k) { return data.data()[i + j * dim[0] + k * dim[0] * dim[1]]; }
		Tensor& data;
		dim1<3> dim;
	};

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
			float tmp[och];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					std::fill(tmp, tmp + och, 0);
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx k = 0; k < och; k++) {
								tmp[k] += a(0, j + m, i + l) * b(k, m, l);
							}
						}
					}
					std::copy(tmp, tmp + och, &c(0, p, o));
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
			float tmp[och];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					std::fill(tmp, tmp + och, 0);
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
					std::copy(tmp, tmp + och, &c(0, p, o));
				}
			}
		}
	}
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
			float tmp[och];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					std::fill(tmp, tmp + och, 0);
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx k = 0; k < bch; k++) {
								tmp[k] += a(0, j + m, i + l) * b(k, m, l);
							}
						}
					}
					std::copy(tmp, tmp + och, &c(0, p, o));
				}
			}
		}
		else if(bch <= 1) {
			float tmp[och];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					std::fill(tmp, tmp + och, 0);
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx n = 0; n < ach; n++) {
								tmp[n] += a(n, j + m, i + l) * b(0, m, l);
							}
						}
					}
					std::copy(tmp, tmp + och, &c(0, p, o));
				}
			}
		}
		else {
			float tmp[och];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					std::fill(tmp, tmp + och, 0);
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
					std::copy(tmp, tmp + och, &c(0, p, o));
				}
			}
		}
	}
	inline void acc_convolve(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa, float mult) {
		idx ach = a.dimension(0);
		idx bch = b.dimension(0);
		idx och = ach * bch;
		if(ach <= 1 && bch <= 1) {
			if(st[0] <= 1 && st[1] <= 1) {
				dim2<2> pad{ shp2{ pa[0], pa[0]}, shp2{ pa[1], pa[1] } };
				c.data.reshape(c.dim).chip(0, 0) += a.chip(0, 0).pad(pad).convolve(b.chip(0, 0), dim1<2>{0, 1})* mult;
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
						c(0, p, o) += tmp * mult;
					}
				}
			}
		}
		else if(ach <= 1) {
			float tmp[och];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					std::fill(tmp, tmp + och, 0);
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
						c(ch, p, o) += tmp[ch] * mult;
				}
			}
		}
		else if(bch <= 1) {
			float tmp[och];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					std::fill(tmp, tmp + och, 0);
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
						c(ch, p, o) += tmp[ch] * mult;
				}
			}
		}
		else {
			float tmp[och];
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					std::fill(tmp, tmp + och, 0);
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
						c(ch, p, o) += tmp[ch] * mult;
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
	inline void acc_mul(const TensorBase<derived>& res, const Tensor& a, const Tensor& b, float mult, shp2 dims = { 1, 0 }) {
		auto& c = const_cast<Eigen::TensorBase<derived>&>(res);
		for(idx i = 0; i < a.dimension(0); i++) {
			for(idx j = 0; j < b.dimension(0); j++) {
				c.chip(i * b.dimension(0) + j, 0) += mult * a.chip(i, 0).contract(b.chip(j, 0), dim2<1>{ dims });
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

	template <class T>
	inline void printnp(const T& t) {
		print("Tensor");
		for(idx i = 0; i < t.NumDimensions; i++) {
			print(t.dimension(i));
		}
		print("\n");
		println(t.format(Eigen::TensorIOFormat::Numpy()));
		print("\n-------------------------------\n");
	}
	template <class T, class ...Ts>
	inline void printnp(const T& t, Ts... ts) {
		printnp(t);
		printnp(ts...);
	}
}



