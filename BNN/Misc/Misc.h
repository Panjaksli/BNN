#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <cmath>
#include <syncstream>
//stupid C++ doesnt provide a way for automatic spacing in ostream...
#define SPC <<' '<<
namespace BNN {
	using std::vector;
	using std::min;
	using std::max;
	template <class T>
	inline T clamp(const T& x, const T& _min, const T& _max) {
		return min(max(x, _min), _max);
	}
	template <class T>
	inline T lerp(const T& x, const T& y, const auto t) {
		return (1 - t) * x + t * y;
	}
	using uchar = uint_fast8_t;
	using uint = uint_fast32_t;
	inline uint xorshift32() {
		thread_local static uint x = 0x6f9f;
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;
		return x;
	}
	inline uint fastrand() {
		thread_local static uint x = 0x6f9f;
		x = (214013U * x + 2531011U);
		return x;
	}

	inline float rafl() {
		uint x = 0x3f800000 | (xorshift32() & 0x007FFFFF);
		return *(float*)&x - 1.f;
	}

	inline float rafl(float min, float max) {
		return rafl() * (max - min) + min;
	}
	inline int raint(int min, int max) {
		return int(rafl() * int(max - min + 1)) + min;
	}
	template <typename T>
	T saturate(const T& x) {
		return min(max(x, T(0)), T(1));
	}
	template <class T = int>
	inline vector<T> shuffled(int n) {
		vector<T> shuff(n);
		for(int i = 0; i < n; i++)
			shuff[i] = i;
		for(int i = 0; i < n - 1; i++)
			std::swap(shuff[i], shuff[raint(i, n - 1)]);
		return shuff;
	}

	inline double timer() {
		auto t = std::chrono::high_resolution_clock::now();
		return std::chrono::duration<double>(t.time_since_epoch()).count();
	}
	inline double timer(double t1) {
		return timer() - t1;
	}
#define cout std::osyncstream(std::cout)
	template <class T>
	inline void print(T t) {
		cout << t << ' ';
	}
	template <class T, class ...Ts>
	inline void print(T t, Ts... ts) {
		print(t);
		print(ts...);
	}
	template <class T>
	inline void printr(T t) {
		cout << t << "\r";
	}
	template <class T, class ...Ts>
	inline void printr(T t, Ts... ts) {
		print(t);
		printr(ts...);
	}
	template <class T>
	inline void println(T t) {
		cout << t << "\n";
	}
	template <class T, class ...Ts>
	inline void println(T t, Ts... ts) {
		print(t);
		println(ts...);
	}
	template <class T>
	inline void printlns(T t) {
		println(t);
	}
	template <class T, class ...Ts>
	inline void printlns(T t, Ts... ts) {
		println(t);
		printlns(ts...);
	}
#undef cout
}