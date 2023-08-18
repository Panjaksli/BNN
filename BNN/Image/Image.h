#pragma once
#include "../Misc/Eigen_util.h"
namespace BNN {
	class Image {
	public:
		Image() {}
		Image(const std::string& name, idx nch, bool print = 0, bool landscape = 0) {
			std::string rename;
			if(load(name, rename, nch) || load(name + ".png", rename, nch) || load(name + ".jpg", rename, nch)
				|| load(name + ".jpeg", rename, nch) || load(name + ".gif", rename, nch) || load(name + ".hdr", rename, nch)
				|| load(name + ".bmp", rename, nch) || load(name + ".tga", rename, nch) || load(name + ".pic", rename, nch)
				|| load(name + ".ppm", rename, nch) || load(name + ".pgm", rename, nch) || load(name + ".psd", rename, nch)) {
				if(landscape && h > w) rotate();
				if(print) println("Loaded image:", rename, "H W C:", h, w, n); return;

			}
			else println("Image was not found:", name);
		}
		~Image() { if(data) free(data); }
		Tensor tensor() const;
		Image(const Tensor& in);
		Image(const Image& cpy) : data((uchar*)malloc(cpy.size())), h(cpy.h), w(cpy.w), n(cpy.n) {
			for(idx i = 0; i < w * h * n; i++)
				data[i] = cpy.data[i];
		}
		const Image& operator=(Image cpy) {
			swap(*this, cpy);
			return *this;
		}
		void rotate();
		//h w d
		inline const uchar& operator()(idx i, idx j, idx k) const {
			return data[(i * w + j) * n + k];
		}
		//h w d
		inline uchar& operator()(idx i, idx j, idx k) {
			return data[(i * w + j) * n + k];
		}
		inline idx size()const {
			return w * h * n;
		}
		inline dim1<3> dim()const {
			return dim1<3>{n, h, w};
		}
		inline dim1<3> pdim()const {
			return dim1<3>{n, h + h % 2, w + w % 2};
		}
		friend void swap(Image& i1, Image& i2) {
			std::swap(i1.data, i2.data);
			std::swap(i1.w, i2.w);
			std::swap(i1.h, i2.h);
			std::swap(i1.n, i2.n);
		}
		bool load(const std::string& name, std::string& rename, idx nch);
		bool save(const std::string& name) const;
		bool save_even(const std::string& name) const;
		uchar* data = nullptr;
		idx h = 0, w = 0, n = 0;
	};
}
