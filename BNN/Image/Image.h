#pragma once
#include "../Misc/Eigen_util.h"
namespace BNN {
	class Image {
	public:
		Image() {}
		Image(const std::string& name, int nch) {
			std::string rename;
			if(load(name, rename, nch) || load(name + ".png", rename, nch) || load(name + ".jpg", rename, nch)
				|| load(name + ".jpeg", rename, nch) || load(name + ".gif", rename, nch) || load(name + ".hdr", rename, nch)
				|| load(name + ".bmp", rename, nch) || load(name + ".tga", rename, nch) || load(name + ".pic", rename, nch)
				|| load(name + ".ppm", rename, nch) || load(name + ".pgm", rename, nch) || load(name + ".psd", rename, nch)) {
				println("Loaded image:", rename, "D H W:", n, h, w); return;
			}
			else println("Image was not found:", name);
		}
		~Image() { if(data) free(data); }
		Tensor tensor() const;
		Image(const Tensor& in);
		Image(const Image& cpy) : data((uchar*)malloc(cpy.size())), w(cpy.w), h(cpy.h), n(cpy.n) {
			for(int i = 0; i < w * h * n; i++)
				data[i] = cpy.data[i];
		}
		const Image& operator=(Image cpy) {
			swap(*this, cpy);
			return *this;
		}
		//h w d
		inline const uchar& operator()(int i, int j, int k) const {
			return data[(i * w + j) * n + k];
		}
		//h w d
		inline uchar& operator()(int i, int j, int k) {
			return data[(i * w + j) * n + k];
		}
		inline ptrdiff_t size()const {
			return w * h * n;
		}
		inline std::array<ptrdiff_t, 3> dim()const {
			return std::array<ptrdiff_t, 3>{n, h, w};
		}
		inline std::array<ptrdiff_t, 3> pdim()const {
			return std::array<ptrdiff_t, 3>{n, h + h % 2, w + w % 2};
		}
		friend void swap(Image& i1, Image& i2) {
			std::swap(i1.data, i2.data);
			std::swap(i1.w, i2.w);
			std::swap(i1.h, i2.h);
			std::swap(i1.n, i2.n);
		}
		bool load(const std::string& name, std::string& rename, int nch);
		bool save(const std::string& name);
		uchar* data = nullptr;
		int w = 0, h = 0, n = 0;
	};
}
