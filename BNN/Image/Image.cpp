#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "stb_image.h"
#include "stb_image_write.h"
#include "Image.h"
namespace BNN {
	Tensor Image::tensor() const {
		//pad to multiple of 2
		Tensor res(n, w + w % 2, h + h % 2);
		for(idx i = 0; i < h; i++) {
			for(idx j = 0; j < w; j++) {
				for(idx k = 0; k < n; k++) {
					res(k, j, i) = operator()(i, j, k) / 255.f;
				}
			}
		}
		return res;
	}
	Image::Image(const Tensor& in) : data((uchar*)malloc(product(in.dimensions()))), h(in.dimension(2)), w(in.dimension(1)), n(in.dimension(0)) {
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
	void Image::rotate() {
		Image tmp = *this;
		std::swap(tmp.w, tmp.h);
		for(idx i = 0; i < h; i++) {
			for(idx j = 0; j < w; j++) {
				for(idx k = 0; k < n; k++) {
					tmp(j, i, k) = operator()(i, j, k);
				}
			}
		}
		swap(*this, tmp);
	}
	bool Image::save(const std::string& name) const {
		return stbi_write_png(name.c_str(), w, h, n, data, w * n);
	}
	bool Image::save_even(const std::string& name) const {
		return stbi_write_png(name.c_str(), w - w % 2, h - h % 2, n, data, w * n);
	}
}
