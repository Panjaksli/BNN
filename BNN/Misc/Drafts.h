#if 1

inline void aconv1(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
	dim1<2> str{ st[0], st[1] };
	dim2<2> pad{ shp2{ pa[0], pa[0]}, shp2{ pa[1], pa[1] } };
	for(idx i = 0; i < b.dimension(0); i++) { //4
		for(idx j = 0; j < a.dimension(0); j++) { //2
			c.data.reshape(c.dim).chip(i * a.dimension(0) + j, 0) = a.chip(j, 0).pad(pad).convolve(b.chip(i, 0), dim1<2>{0, 1}).stride(str);
		}
	}
}

inline void aconv2(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
	for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
		for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
			for(idx n = 0; n < a.dimension(0); n++) {
				for(idx k = 0; k < b.dimension(0); k++) {
					float tmp = 0;
					for(idx l = 0; l < b.dimension(2); l++) {
						for(idx m = 0; m < b.dimension(1); m++) {
							if(size_t(i + l) < a.dimension(2) && size_t(j + m) < a.dimension(1)) {
								tmp += a(n, j + m, i + l) * b(k, m, l);
							}

						}
					}
					c(k * a.dimension(0) + n, p, o) = tmp;
				}
			}
		}
	}
}



inline void conv1(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
	c.data.setZero();
	idx nch = b.dimension(0) / a.dimension(0);
	dim1<2> str{ st[0], st[1] };
	dim2<2> pad{ shp2{ pa[0], pa[0]}, shp2{ pa[1], pa[1] } };
	for(idx i = 0; i < nch; i++) {
		for(idx j = 0; j < a.dimension(0); j++) {
			c.data.reshape(c.dim).chip(i, 0) += a.chip(j, 0).pad(pad).convolve(b.chip(i * a.dimension(0) + j, 0), dim1<2>{0, 1}).stride(str);
		}
	}
}

inline void conv2(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
	idx nch = b.dimension(0) / a.dimension(0);
	for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
		for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
			for(idx k = 0; k < nch; k++) {
				float tmp = 0;
				for(idx l = 0; l < b.dimension(2); l++) {
					for(idx m = 0; m < b.dimension(1); m++) {
						for(idx n = 0; n < a.dimension(0); n++) {
							tmp += (size_t(i + l) < a.dimension(2) && size_t(j + m) < a.dimension(1) ?
								a(n, j + m, i + l) : 0.f) * b(k * a.dimension(0) + n, m, l);
						}
					}
				}
				c(k, p, o) = tmp;
			}
		}
	}
}

inline void conv3(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
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
					int clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
					int clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
					for(idx l = i < 0 ? -i : 0; l < b.dimension(2) - clip_l; l++) {
						for(idx m = j < 0 ? -j : 0; m < b.dimension(1) - clip_m; m++) {
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
				int clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
				int clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
				for(idx l = i < 0 ? -i : 0; l < b.dimension(2) - clip_l; l++) {
					for(idx m = j < 0 ? -j : 0; m < b.dimension(1) - clip_m; m++) {
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
				int clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
				int clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
				for(idx l = i < 0 ? -i : 0; l < b.dimension(2) - clip_l; l++) {
					for(idx m = j < 0 ? -j : 0; m < b.dimension(1) - clip_m; m++) {
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
				int clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
				int clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
				for(idx l = i < 0 ? -i : 0; l < b.dimension(2) - clip_l; l++) {
					for(idx m = j < 0 ? -j : 0; m < b.dimension(1) - clip_m; m++) {
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

inline void aconv3(Reshape c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
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
					int clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
					int clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
					for(idx l = i < 0 ? -i : 0; l < b.dimension(2) - clip_l; l++) {
						for(idx m = j < 0 ? -j : 0; m < b.dimension(1) - clip_m; m++) {
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
				int clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
				int clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
				for(idx l = i < 0 ? -i : 0; l < b.dimension(2) - clip_l; l++) {
					for(idx m = j < 0 ? -j : 0; m < b.dimension(1) - clip_m; m++) {
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
				int clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
				int clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
				for(idx l = i < 0 ? -i : 0; l < b.dimension(2) - clip_l; l++) {
					for(idx m = j < 0 ? -j : 0; m < b.dimension(1) - clip_m; m++) {
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
				int clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
				int clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
				for(idx l = i < 0 ? -i : 0; l < b.dimension(2) - clip_l; l++) {
					for(idx m = j < 0 ? -j : 0; m < b.dimension(1) - clip_m; m++) {
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

int main() {
	idx i = 2000;
	idx k = 5;
	idx s = 1;
	idx p = 2;
	idx in = 1;
	idx n = 1;
	Tensor x(in, i, i); x.setRandom();
	Tensor w(n * in, k, k); w.setRandom();
	Tensor y1(n, c_dim(i, k, s, p), c_dim(i, k, s, p));
	Tensor y2(n, c_dim(i, k, s, p), c_dim(i, k, s, p));
	Tensor y3(n, c_dim(i, k, s, p), c_dim(i, k, s, p));
	double t; double t1 = 0, t2 = 0, t3 = 0;
	for(int i = 0; i < 1; i++) {
		t = timer();
		conv1(y1, x, w, s, p);
		t1 += timer(t);
		t = timer();
		conv2(y2, x, w, s, p);
		t2 += timer(t);
		t = timer();
		conv3(y3, x, w, s, p);
		t3 += timer(t);
	}
	println(t1, t2, t3);
	//y1 = y2;
	printnp(fsca((y1 - y2).mean()));
	printnp(fsca((y1 - y3).mean()));
}
#else
