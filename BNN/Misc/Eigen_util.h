#pragma once
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "Misc.h"

namespace BNN {
	using idx = int_fast32_t;
	using Eigen::TensorBase;
	using Eigen::TensorRef;

	using Tensor = Eigen::Tensor<float, 3, Eigen::ColMajor, idx>;
	using Tenarr = Eigen::Tensor<float, 4, Eigen::ColMajor, idx>;
	using fsca = Eigen::TensorFixedSize<float, Eigen::Sizes<>, 0, idx>;

	template <size_t N>
	using dim1 = Eigen::DSizes<idx, N>;
	//using shp2 = Eigen::IndexPair<idx>;
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
	struct shp3 : public dim1<3>{
		shp3() {}
		shp3(idx d1) : dim1<3>{ d1,1,1 } {}
		shp3(idx d1, idx d2) : dim1<3>{ d1,d2,1 } {}
		shp3(idx d1, idx d2, idx d3) : dim1<3>{ d1,d2,d3 } {}
		shp3(const dim1<3>& d) : dim1<3>(d) {}
	};
	inline idx product(const dim1<3>& x) { return x[0] * x[1] * x[2]; }
	using shp4 = dim1<4>;
	inline constexpr idx c_dim(idx i, idx k, idx s, idx p) { return (i + 2 * p - k) / s + 1; }
	inline constexpr idx t_dim(idx i, idx k, idx s, idx p) { return (i - 1) * s + k - 2 * p; }

	inline void random_r(Tensor& c, float min = 0.f, float max = 1.f) {
		c.setRandom();
		c = (max - min) * c + min;
	}

	struct TensRef {
		TensRef(Tensor& data) : data(data), dim(data.dimensions()) {}
		TensRef(Tensor& data, shp3 dim) : data(data), dim(dim) {}
		const float& operator() (idx i, idx j, idx k)const { return data.data()[i + j * dim[0] + k * dim[0] * dim[1]]; }
		float& operator() (idx i, idx j, idx k) { return data.data()[i + j * dim[0] + k * dim[0] * dim[1]]; }
		const float& operator[] (idx i)const { return data.data()[i]; }
		float& operator[] (idx i) { return data.data()[i]; }
		const Tensor& tensor()const {
			return data;
		}
		Tensor& tensor() {
			return data;
		}
		const auto reshape()const {
			return data.reshape(dim);
		}
		auto reshape() {
			return data.reshape(dim);
		}
		const auto chip(idx id, idx ax)const {
			return data.reshape(dim).chip(id, ax);
		}
		auto chip(idx id, idx ax) {
			return data.reshape(dim).chip(id, ax);
		}
		void setZero() {
			data.setZero();
		}
		idx dimension(idx i)const { return dim[i]; }
		Tensor& data;
		dim1<3> dim;
	};
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



