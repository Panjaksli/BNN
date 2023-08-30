#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "stb_image.h"
#include "stb_image_write.h"
#include "Image.h"
namespace BNN {
	Tensor Image::tensor_rgb(bool even) const {
		idx tw = w - even * (w % 2);
		idx th = h - even * (h % 2);
		Tensor res(n, tw, th);
		for(idx i = 0; i < th; i++) {
			for(idx j = 0; j < tw; j++) {
				for(idx k = 0; k < n; k++) {
					res(k, j, i) = operator()(i, j, k) / 255.f;
				}
			}
		}
		return res;
	}
	Tensor Image::tensor_yuv(bool even) const {
		idx tw = w - even * (w % 2);
		idx th = h - even * (h % 2);
		float rgb[16]{};
		Tensor res(n, tw, th);
		for(idx i = 0; i < th; i++) {
			for(idx j = 0; j < tw; j++) {
				if(n < 3) {
					res(0, j, i) = operator()(i, j, 0) / 255.f;
				}
				else {
					for(idx k = 0; k < n; k++) {
						rgb[k] = operator()(i, j, k) / 255.f;
					}
					res(0, j, i) = 0.299f * rgb[0] + 0.587f * rgb[1] + 0.114 * rgb[2];
					res(1, j, i) = -0.147f * rgb[0] - 0.289f * rgb[1] + 0.436 * rgb[2];
					res(2, j, i) = 0.615f * rgb[0] - 0.515f * rgb[1] - 0.1f * rgb[2];
				}
			}
		}
		return res;
	}
	Image::Image(const Tensor& in) : data((uchar*)malloc(product(in.dimensions()))), n(in.dimension(0)), w(in.dimension(1)), h(in.dimension(2)) {
		Tensor tmp = in.clip(0.f, 1.f) * 255.f + 0.5f;
		for(idx i = 0; i < h; i++) {
			for(idx j = 0; j < w; j++) {
				for(idx k = 0; k < n; k++) {
					operator()(i, j, k) = tmp(k, j, i);
				}
			}
		}
	}
	bool Image::load(const std::string& name, std::string& rename, idx nch) {
		idx tw, th, tn;
		uchar* tmp = stbi_load(name.c_str(), &tw, &th, &tn, nch);
		if(tmp) {
			rename = name;
			data = tmp;
			w = tw;
			h = th;
			n = nch;
			return true;
		}
		return false;
	}
	Image& Image::resize(int _w, int _h, Interpol filter) {
		Image tmp(n, _w, _h);
		float s1 = tmp.w > 0 ? float(w) / (tmp.w) : 0;
		float s2 = tmp.h > 0 ? float(h) / (tmp.h) : 0;
		if(filter == Nearest) {
			for(idx i = 0; i < tmp.h; i++) {
				idx li = (i + 0.5f) * s2;
				for(idx j = 0; j < tmp.w; j++) {
					idx lj = (j + 0.5f) * s1;
					for(idx k = 0; k < n; k++) {
						tmp(i, j, k) = operator()(li, lj, k);
					}
				}
			}
		}
		else if(filter == Linear) {
			for(idx i = 0; i < tmp.h; i++) {
				float fi = fmaxf((i + 0.5f) * s2 - 0.5f, 0);
				idx li = fi;
				idx hi = min(li + 1, h - 1);
				float wi = fi - li;
				for(idx j = 0; j < tmp.w; j++) {
					float fj = fmaxf((j + 0.5f) * s1 - 0.5f, 0);
					idx lj = fj;
					idx hj = min(lj + 1, w - 1);
					float wj = fj - lj;
					for(idx k = 0; k < n; k++) {
						float a = operator()(li, lj, k);
						float b = operator()(li, hj, k);
						float c = operator()(hi, lj, k);
						float d = operator()(hi, hj, k);
						tmp(i, j, k) = lerp(lerp(a, b, wj), lerp(c, d, wj), wi);
					}
				}
			}
		}
		else if(filter == Cubic) {
			for(idx i = 0; i < tmp.h; i++) {
				float y = fmaxf((i + 0.5f) * s2 - 0.5f, 0);
				int y0 = y;
				float yd = y - y0;
				for(idx j = 0; j < tmp.w; j++) {
					float x = fmaxf((j + 0.5f) * s1 - 0.5f, 0);
					int x0 = x;
					float xd = x - x0;
					for(idx k = 0; k < n; k++) {
						float p[16];
						for(int ii = 0; ii < 4; ii++) {
							for(int jj = 0; jj < 4; jj++) {
								p[ii * 4 + jj] = operator()(clamp(y0 + ii - 1, 0, h - 1), clamp(x0 + jj - 1, 0, w - 1), k);
							}
						}
						tmp(i, j, k) = clamp(bicerp(p, xd, yd), 0.f, 255.f);
					}
				}
			}
		}
		swap(*this, tmp);
		return *this;
	}

	Image& Image::rotate() {
		Image tmp(n, h, w);
		for(idx i = 0; i < h; i++) {
			for(idx j = 0; j < w; j++) {
				for(idx k = 0; k < n; k++) {
					tmp(j, i, k) = operator()(i, j, k);
				}
			}
		}
		swap(*this, tmp);
		return *this;
	}
	bool Image::save(const std::string& name) const {
		return stbi_write_png(name.c_str(), w, h, n, data, w * n);
	}
	bool Image::save_jpg(const std::string& name) const {
		return stbi_write_jpg(name.c_str(), w, h, n, data, 90);
	}
	bool Image::save_even(const std::string& name) const {
		return stbi_write_png(name.c_str(), w - w % 2, h - h % 2, n, data, w * n);
	}
}
