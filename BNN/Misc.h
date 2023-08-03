#pragma once
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <cmath>

#include <unsupported/Eigen/CXX11/Tensor>
using std::vector;
using std::min;
using std::max;
using std::clamp;

using uint = uint32_t;

template <typename T, class deleter = std::default_delete<T>>
using uptr = std::unique_ptr<T, deleter>;

template <typename T>
using sptr = std::shared_ptr<T>;

template <typename T>
auto news = &std::make_shared<T>;

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
	return rafl(min, max);
}
template <typename T>
T saturate(const T& x) {
	return min(max(x, T(0)), T(1));
}
template <class T = int>
inline vector<T> shuffled(int n) {
	vector<T> shuff(n);
	for (int i = 0; i < n; i++)
		shuff[i] = i;
	for (int i = 0; i < n - 1; i++)
		std::swap(shuff[i], shuff[raint(i, n)]);
	return shuff;
}

inline double timer() {
	auto t = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double>(t.time_since_epoch()).count();
}
inline double timer(double t1) {
	auto t = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double>(t.time_since_epoch()).count() - t1;
}

template <class T>
inline void print(T t) {
	std::cout << t << ' ';
}
template <class T, class ...Ts>
inline void print(T t, Ts... ts) {
	print(t);
	print(ts...);
}
template <class T>
inline void println(T t) {
	std::cout << t << '\n';
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

