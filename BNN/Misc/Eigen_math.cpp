#include "Eigen_util.h"
namespace BNN {
	constexpr int CACHE_SIZE = 4096;
	//Classic convolution of N input channels with N*K filters

	void convolve(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
		idx ich = a.dimension(0);
		idx och = b.dimension(0) / ich;
		if(och > CACHE_SIZE) return;
		alignas(64) float tmp[CACHE_SIZE];
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
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, och * sizeof(float));
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx k = 0; k < och; k++) {
								tmp[k] += a(0, j + m, i + l) * b(k, m, l);
							}
						}
					}
					memcpy(&c(0, p, o), tmp, och * sizeof(float));
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
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, och * sizeof(float));
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
					memcpy(&c(0, p, o), tmp, och * sizeof(float));
				}
			}
		}
	}
	//fixing a fuckup where I was convolving incorrect filters in backprop, also reverse operation is included inhouse
	//stupid me
	//It was convolving for example 4 output filters by 4 first filters of the kernel which werent the filters corresponding to the input !!!

	void rev_convolve(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
		idx ich = a.dimension(0);
		idx och = b.dimension(0) / ich;
		idx bdm = b.dimension(1);
		idx bdl = b.dimension(2);
		if(och > CACHE_SIZE) return;
		alignas(64) float tmp[CACHE_SIZE];
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
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - bdl + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - bdm + 1); j += st[0], p++) {
					memset(tmp, 0, och * sizeof(float));
					idx clip_l = max(0, i + bdl - a.dimension(2));
					idx clip_m = max(0, j + bdm - a.dimension(1));
					for(idx l = max(-i, 0); l < bdl - clip_l; l++) {
						for(idx m = max(-j, 0); m < bdm - clip_m; m++) {
							for(idx k = 0; k < och; k++) {
								tmp[k] += a(0, j + m, i + l) * b(k, bdm - 1 - m, bdl - 1 - l);
							}
						}
					}
					memcpy(&c(0, p, o), tmp, och * sizeof(float));
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
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - bdl + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - bdm + 1); j += st[0], p++) {
					memset(tmp, 0, och * sizeof(float));
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
					memcpy(&c(0, p, o), tmp, och * sizeof(float));
				}
			}
		}
	}
	//Convolve all filter combinations, N channels, K filters, N*K output channels

	void all_convolve(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
		idx ach = a.dimension(0);
		idx bch = b.dimension(0);
		idx och = ach * bch;
		if(och > CACHE_SIZE) return;
		alignas(64) float tmp[CACHE_SIZE];
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
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, och * sizeof(float));
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx k = 0; k < bch; k++) {
								tmp[k] += a(0, j + m, i + l) * b(k, m, l);
							}
						}
					}
					memcpy(&c(0, p, o), tmp, och * sizeof(float));
				}
			}
		}
		else if(bch <= 1) {
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, och * sizeof(float));
					idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
					idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
					for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
						for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
							for(idx n = 0; n < ach; n++) {
								tmp[n] += a(n, j + m, i + l) * b(0, m, l);
							}
						}
					}
					memcpy(&c(0, p, o), tmp, och * sizeof(float));
				}
			}
		}
		else {
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, och * sizeof(float));
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
					memcpy(&c(0, p, o), tmp, och * sizeof(float));
				}
			}
		}
	}
	//Convolve all combinations and accumulate to output

	void acc_convolve(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
		idx ach = a.dimension(0);
		idx bch = b.dimension(0);
		idx och = ach * bch;
		if(och > CACHE_SIZE) return;
		alignas(64) float tmp[CACHE_SIZE];
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
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, och * sizeof(float));
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
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, och * sizeof(float));
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
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
					memset(tmp, 0, och * sizeof(float));
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
	//Convolve each input with each channel

	void convolve_1to1(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
		idx ich = a.dimension(0);
		if(ich > CACHE_SIZE) return;
		alignas(64) float tmp[CACHE_SIZE];
		for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
			for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
				memset(tmp, 0, ich * sizeof(float));
				idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
				idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
				for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
					for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
						for(idx n = 0; n < ich; n++) {
							tmp[n] += a(n, j + m, i + l) * b(n, m, l);
						}
					}
				}
				memcpy(&c(0, p, o), tmp, ich * sizeof(float));
			}
		}
	}

	void rev_convolve_1to1(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
		idx ich = a.dimension(0);
		if(ich > CACHE_SIZE) return;
		alignas(64) float tmp[CACHE_SIZE];
		for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
			for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
				memset(tmp, 0, ich * sizeof(float));
				idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
				idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
				for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
					for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {

						for(idx n = 0; n < ich; n++) {
							tmp[n] += a(n, j + m, i + l) * b(n, b.dimension(1) - m - 1, b.dimension(2) - l - 1);
						}
					}
				}
				memcpy(&c(0, p, o), tmp, ich * sizeof(float));
			}
		}
	}

	void acc_convolve_1to1(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
		idx ich = a.dimension(0);
		if(ich > CACHE_SIZE) return;
		alignas(64) float tmp[CACHE_SIZE];
		for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
			for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
				memset(tmp, 0, ich * sizeof(float));
				idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
				idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
				for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
					for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
						for(idx n = 0; n < ich; n++) {
							tmp[n] += a(n, j + m, i + l) * b(n, m, l);
						}
					}
				}
				for(idx ch = 0; ch < ich; ch++)
					c(ch, p, o) += tmp[ch];
			}
		}
	}
	//bilinear resize
	void resize_r(Reshape y, const Tensor& x, Interpol filter) {
		if(y.dim[0] != x.dimension(0) || y.dim[0] > CACHE_SIZE) return;
		alignas(64) float tmp[CACHE_SIZE];
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
					memcpy(&y(0, j, i), tmp, y.dim[0] * sizeof(float));
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
					memcpy(&y(0, j, i), tmp, y.dim[0] * sizeof(float));
				}
			}
		}
		else if(filter == Cubic) {
			for(idx i = 0; i < y.dim[2]; i++) {
				float yc = max((i + 0.5f) * s2 - 0.5f, 0.f);
				int y0 = yc;
				float yd = yc - y0;
				for(idx j = 0; j < y.dim[1]; j++) {
					float xc = max((j + 0.5f) * s1 - 0.5f, 0.f);
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
					memcpy(&y(0, j, i), tmp, y.dim[0] * sizeof(float));
				}
			}
		}
	}
	//multiply all matrix combinations stored as a0b0,a0b1,a1b0,a1b1....
	
}