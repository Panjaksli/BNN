#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "stb_image.h"
#include "stb_image_write.h"
#include "Image.h"
namespace BNN {
	Tensor Image::tensor_rgb() const {
		Tensor res(n, w, h);
		for(idx i = 0; i < h; i++) {
			for(idx j = 0; j < w; j++) {
				for(idx k = 0; k < n; k++) {
					res(k, j, i) = operator()(i, j, k) / 255.f;
				}
			}
		}
		return res;
	}
	Tensor Image::tensor_yuv() const {
		float rgb[16]{};
		Tensor res(n, w, h);
		for(idx i = 0; i < h; i++) {
			for(idx j = 0; j < w; j++) {
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
	Image& Image::resize(int _w, int _h) {
		Image tmp(n, _w, _h);
		float s1 = tmp.w > 1 ? (w - 1) / (tmp.w - 1.f) : 0;
		float s2 = tmp.h > 1 ? (h - 1) / (tmp.h - 1.f) : 0;
		for(idx i = 0; i < tmp.h; i++) {
			idx li = i * s2;
			idx hi = min(li + 1, h - 1);
			float wi = i * s2 - li;
			for(idx j = 0; j < tmp.w; j++) {
				idx lj = j * s1;
				idx hj = min(lj + 1, w - 1);
				float wj = j * s1 - lj;
				for(idx k = 0; k < n; k++) {
					float a = operator()(li, lj, k);
					float b = operator()(li, hj, k);
					float c = operator()(hi, lj, k);
					float d = operator()(hi, hj, k);
					tmp(i,j,k) = (1.f - wi) * (a * (1.f - wj) + b * wj) + wi * (c * (1.f - wj) + d * wj);
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
