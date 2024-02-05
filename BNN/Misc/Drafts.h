#if 1
using Eigen::TensorMap;
using Eigen::TensorRef;
struct Transform {
	Transform(Tensor& data) : data(data), dim(data.dimensions()) {
		compute_dim();
	}
	Transform(Tensor& data, shp3 dim, shp3 dil, shp3 pad = shp3(0, 0, 0)) : data(data), dim(dim), dil(dil), pad(pad) {
		compute_dim();
	}
	float operator() (idx i, idx j, idx k)const {
		if(i % dil[0] || j % dil[1] || k % dil[2]) return 0.f;
		else return data.data()[i / dil[0] + j / dil[1] * dim[0] + k / dil[2] * dim[0] * dim[1]];
	}
	float& operator() (idx i, idx j, idx k) {
		if(i % dil[0] || j % dil[1] || k % dil[2]) return none;
		else return data.data()[i / dil[0] + j / dil[1] * dim[0] + k / dil[2] * dim[0] * dim[1]];
	}
	inline auto reshape() { return data.reshape(dim); }
	inline auto inflate() { return data.inflate(dil); }
	inline auto reshape_inflate() { return data.reshape(dim).inflate(dil); }
	inline idx dimension(idx i) const { return odim[i]; }
	inline const dim1<3>& dimensions(idx i) const { return odim; }
	inline void compute_dim() {
		for(int i = 0; i < 3; i++)
			odim[i] = (dim[i] - 1) * dil[i] + 1;
	}
	Tensor& data;
	dim1<3> odim, dim, dil = dim1<3>{ 1,1,1 }, pad = dim1<3>{ 0,0,0 };
	float none = 0.f;
};
int main() {
	{
		Tensor x(1, 7, 7);
		Tensor y(1, 4, 4);
		y.setRandom();
		Transform z(y, y.dimensions(), shp3(1, 3, 3));
		for(int i = 0; i < x.dimension(2); i++) {
			for(int j = 0; j < x.dimension(1); j++) {
				for(int k = 0; k < x.dimension(0); k++) {
					x(k, j, i) = z(k, j, i);
				}
			}
		}
		printnp(x);
		return 0;
	}
void pp1(Tensor& x, const Tensor& y, const Tensor& z) {
	for(int i = 0; i < x.size(); i++) {
		x(i, 0, 0) = y(i, 0, 0) * z(i, 0, 0);
	}
}
void pp2(Tensor& x, Reshape y, Reshape z) {
	for(int i = 0; i < x.size(); i++) {
		x(i, 0, 0) = y(i, 0, 0) * z(i, 0, 0);
	}
}
int main() {
	Tensor x(1000000, 1, 1);
	Tensor y(100, 100, 100);
	Tensor z(100, 100, 100);
	y.setRandom();
	z.setRandom();
	double t = 0;
	auto yr = y.reshape(dim1<3>{x.size(), 1, 1});
	auto zr = z.reshape(dim1<3>{x.size(), 1, 1});
	t = timer();
	pp1(x, yr, zr);
	println(timer(t));
	t = timer();
	pp2(x, { y ,x.dimensions() }, { z ,x.dimensions() });
	println(timer(t));
inline float sum(__m256 x) {
	__m256 y = x + _mm256_permute_ps(x, 0b00011011); // 41 32 23 14
	__m256 z = y + _mm256_permute_ps(y, 0b01000001);
	return z[0] + z[4];
}
inline float dot_simd(const float* vec1, const float* vec2, int n) {
	float y = 0;
	__m256 vy = _mm256_setzero_ps();
	for(int i = 0; i < n; i += 8) {
		if(i + 8 <= n) {
			__m256 a = _mm256_loadu_ps(&vec1[i]);
			__m256 b = _mm256_loadu_ps(&vec2[i]);
			vy += a * b;
		}
		else {
			for(int j = i; j < n; j++)
				y += vec1[j] * vec2[j];
		}
	}
	return y + sum(vy);
}
using Rensor = Eigen::Tensor<float, 3, Eigen::RowMajor, int>;

inline void rm_convolve(Rensor& c, const Rensor& a, const Rensor& b, shp2 st, shp2 pa) {
	idx ich = a.dimension(0);
	idx och = b.dimension(0) / ich;
	/*c.setZero();
	for(int i = 0; i < och; i++) {
		for(int j = 0; j < ich; j++) {
			c.chip(i, 0) += a.chip(j, 0).pad(dim2<2>{shp2(pa[0], pa[0]), shp2(pa[1], pa[1])}).convolve(b.chip(i * ich + j, 0), dim1<2>{0, 1}).stride(dim1<2>{st[0], st[1]});
		}
	}
	return;*/


	for(idx k = 0; k < och; k++) {
		for(idx j = -pa[0], p = 0; j < (a.dimension(1) + pa[0] - b.dimension(1) + 1); j += st[0], p++) {
			for(idx i = -pa[1], o = 0; i < (a.dimension(2) + pa[1] - b.dimension(2) + 1); i += st[1], o++) {
				float tmp = 0;
				idx clip_l = max(0, i + b.dimension(2) - a.dimension(2));
				idx clip_m = max(0, j + b.dimension(1) - a.dimension(1));
				for(idx n = 0; n < ich; n++) {
					for(idx m = max(-j, 0); m < b.dimension(1) - clip_m; m++) {
						for(idx l = max(-i, 0); l < b.dimension(2) - clip_l; l++) {
							tmp += a(n, j + m, i + l) * b(k * ich + n, m, l);
						}
					}
				}
				c(k, p, o) = tmp;
			}

		}
	}
}

inline void cm_convolve(Tensor& c, const Tensor& a, const Tensor& b, shp2 st, shp2 pa) {
	idx ich = a.dimension(0);
	idx och = b.dimension(0) / ich;
	for(int i = 0; i < och; i++) {
		c.chip(i, 0).setZero();
		for(int j = 0; j < ich; j++) {
			c.chip(i, 0) += a.chip(j, 0).pad(dim2<2>{shp2(pa[0], pa[0]), shp2(pa[1], pa[1])}).convolve(b.chip(i * ich + j, 0), dim1<2>{0, 1}).stride(dim1<2>{st[0], st[1]});
		}
	}
}
struct TensorOP {
	TensorOP(Tensor& data) : data(data), dim(data.dimensions()) {}
	TensorOP(Tensor& data, shp3 dim, shp3 st = shp3(1, 1, 1), shp3 pa = shp3(0, 0, 0), shp3 in = shp3(0, 0, 0)) : data(data), dim(dim), st(st) {}
	operator Tensor& () { return data; }
	operator const Tensor& () const { return data; }
	auto operator()() { return data.reshape(dim).inflate(in).pad(dim2<3>{shp2(pa[0]), shp2(pa[1]), shp2(pa[2])}).stride(st); }
	const auto operator()() const { return data.reshape(dim).inflate(in).pad(dim2<3>{shp2(pa[0]), shp2(pa[1]), shp2(pa[2])}).stride(st); }
	const float& operator() (idx i, idx j, idx k)const { return data.data()[i * st[0] + j * dim[0] * st[1] + k * dim[0] * dim[1] * st[2]]; }
	float& operator() (idx i, idx j, idx k) { return data.data()[i * st[0] + j * dim[0] * st[1] + k * dim[0] * dim[1] * st[2]]; }
	Tensor& data;
	dim1<3> dim;
	dim1<3> st{ 1,1,1 };
	dim1<3> pa{ 0,0,0 };
	dim1<3> in{ 1,1,1 };
};


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
					idx clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
					idx clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
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
				idx clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
				idx clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
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
				idx clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
				idx clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
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
				idx clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
				idx clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
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
					idx clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
					idx clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
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
				idx clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
				idx clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
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
				idx clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
				idx clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
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
				idx clip_l = (i + b.dimension(2) > a.dimension(2)) * (i + b.dimension(2) - a.dimension(2));
				idx clip_m = (j + b.dimension(1) > a.dimension(1)) * (j + b.dimension(1) - a.dimension(1));
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

idx main() {
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
	for(idx i = 0; i < 1; i++) {
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

template <typename TensorType>
void conv2d1(TensorType& output, const TensorType& input, const Tenarr& kernel, shp2 stride = 1, shp2 dilate = 1, Eigen::PaddingType padding_type = Eigen::PADDING_SAME) {
	const auto NumDims = input.NumDimensions;
	const auto kernelFilters = kernel.dimension(0);
	const auto kernelChannels = kernel.dimension(1);
	const auto kernelRows = kernel.dimension(2);
	const auto kernelCols = kernel.dimension(3);
	const auto kernel_dims = dim1<2>{ kernelFilters, kernelChannels * kernelRows * kernelCols };
	const auto InputRows = input.dimension(1);
	const auto InputCols = input.dimension(2);
	const auto kernelRowsEff = kernelRows + (kernelRows - 1) * (dilate[0] - 1);
	const auto kernelColsEff = kernelCols + (kernelCols - 1) * (dilate[0] - 1);
	auto out_height = 0;
	auto out_width = 0;
	switch(padding_type) {
		case Eigen::PADDING_VALID:
			out_height = ceil((InputRows - kernelRowsEff + 1.f) / static_cast<float>(stride[0]));
			out_width = ceil((InputCols - kernelColsEff + 1.f) / static_cast<float>(stride[1]));
			break;
		case Eigen::PADDING_SAME:
			out_height = ceil(InputRows / static_cast<float>(stride[0]));
			out_width = ceil(InputCols / static_cast<float>(stride[1]));
			break;
		default:
			out_height = 0;
			out_width = 0;
			eigen_assert(false && "unexpected padding");
	}
	dim1<2> pre_contract_dims;
	pre_contract_dims[0] = kernelChannels * kernelRows * kernelCols;
	pre_contract_dims[1] = out_height * out_width;
	for(int i = 3; i < NumDims; ++i) {
		pre_contract_dims[1] *= input.dimension(i);
	}
	dim1<3> post_contract_dims;
	post_contract_dims[0] = kernelFilters;
	post_contract_dims[1] = out_height;
	post_contract_dims[2] = out_width;
	for(int i = 3; i < NumDims; ++i) {
		post_contract_dims[i] = input.dimension(i);
	}
	dim2<1> contract_dims{ shp2(1, 0) };
	auto patches = input.extract_image_patches(kernelRows, kernelCols, stride[0], stride[1], dilate[0], dilate[1], padding_type).reshape(pre_contract_dims);
	output = kernel.reshape(kernel_dims).contract(patches, contract_dims).reshape(post_contract_dims);
}

template <typename TensorType>
void conv2d2(TensorType& output, const TensorType& input, const Tenarr& kernel, shp2 stride = 1, shp2 dilate = 1, Eigen::PaddingType padding_type = Eigen::PADDING_SAME) {
	const auto NumDims = input.NumDimensions;
	const auto kernelFilters = kernel.dimension(3);
	const auto kernelChannels = kernel.dimension(0);
	const auto kernelRows = kernel.dimension(1);
	const auto kernelCols = kernel.dimension(2);
	const auto kernel_dims = dim1<2>{ kernelChannels * kernelRows * kernelCols, kernelFilters };
	const auto InputRows = input.dimension(1);
	const auto InputCols = input.dimension(2);
	const auto kernelRowsEff = kernelRows + (kernelRows - 1) * (dilate[0] - 1);
	const auto kernelColsEff = kernelCols + (kernelCols - 1) * (dilate[0] - 1);
	auto out_height = 0;
	auto out_width = 0;
	switch(padding_type) {
		case Eigen::PADDING_VALID:
			out_height = ceil((InputRows - kernelRowsEff + 1.f) / static_cast<float>(stride[0]));
			out_width = ceil((InputCols - kernelColsEff + 1.f) / static_cast<float>(stride[1]));
			break;
		case Eigen::PADDING_SAME:
			out_height = ceil(InputRows / static_cast<float>(stride[0]));
			out_width = ceil(InputCols / static_cast<float>(stride[1]));
			break;
		default:
			out_height = 0;
			out_width = 0;
			eigen_assert(false && "unexpected padding");
	}
	dim1<2> pre_contract_dims;
	pre_contract_dims[0] = kernelChannels * kernelRows * kernelCols;
	pre_contract_dims[1] = out_height * out_width;
	for(int i = 3; i < NumDims; ++i) {
		pre_contract_dims[1] *= input.dimension(i);
	}
	dim1<3> post_contract_dims;
	post_contract_dims[0] = kernelFilters;
	post_contract_dims[1] = out_height;
	post_contract_dims[2] = out_width;
	for(int i = 3; i < NumDims; ++i) {
		post_contract_dims[i] = input.dimension(i);
	}
	dim2<1> contract_dims{ shp2(0, 0) };
	auto patches = input.extract_image_patches(kernelRows, kernelCols, stride[0], stride[1], dilate[0], dilate[1], padding_type).reshape(pre_contract_dims);
	output = kernel.reshape(kernel_dims).contract(patches, contract_dims).reshape(post_contract_dims);

}