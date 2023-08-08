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
		Tensor res(n, h + h % 2, w + w % 2);
		for(int j = 0; j < w; j++) {
			for(int i = 0; i < h; i++) {
				for(int k = 0; k < n; k++) {
					res(k, i, j) = operator()(i, j, k) / 255.f;
				}
			}
		}
		return res;
	}
	Image::Image(const Tensor& in) : data((uchar*)malloc(product(in.dimensions()))), w(in.dimension(2)), h(in.dimension(1)), n(in.dimension(0)) {
		Tensor tmp = in.clip(0.f,1.f) * 255.f + 0.5f;
		for(int j = 0; j < w; j++) {
			for(int i = 0; i < h; i++) {
				for(int k = 0; k < n; k++) {
					operator()(i, j, k) = tmp(k, i, j);
				}
			}
		}
	}
	bool Image::load(const std::string& name, std::string& rename, int nch) {
		int tw, th, tn;
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
	bool Image::save(const std::string& name) {
		return stbi_write_png(name.c_str(), w, h, n, data, w * n);
	}
}
