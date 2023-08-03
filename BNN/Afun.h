#pragma once
#include "Misc.h"
namespace BNN {
	class Efun {
	public:
		enum Type { t_mse, t_mae } type = t_mse;
		Efun(){}
		Efun(Type type) : type(type) {}
	}; 
	class Afun {
	public:
		enum Type { t_lin, t_relu, t_lrelu, t_sat, t_sig, t_clu, t_swish, t_tanh } type = t_relu;
		Afun() {}
		Afun(Type type) : type(type) {}
		inline auto fx() const{
			switch (type) {
			case t_lin: return lin::fx;
			case t_relu:return relu::fx;
			case t_lrelu:return lrelu::fx;
			case t_sat:return sat::fx;
			case t_sig:return sig::fx;
			case t_clu:return clu::fx;
			case t_swish:return swish::fx;
			case t_tanh:return tanh::fx;
			default:return lin::fx;
			}
		}
		inline auto dx() const{
			switch (type) {
			case t_lin: return lin::dx;
			case t_relu:return relu::dx;
			case t_lrelu:return lrelu::dx;
			case t_sat:return sat::dx;
			case t_sig:return sig::dx;
			case t_clu:return clu::dx;
			case t_swish:return swish::dx;
			case t_tanh:return tanh::dx;
			default:return lin::dx;
			}
		}
		struct lin {
			static float fx(float x) { return x; }
			static float dx(float x) { return 1.f; }
			static constexpr Type type = t_lin;
		};
		struct relu {
			static float fx(float x) { return max(x, 0.f); }
			static float dx(float x) { return (x > 0.f); }
			static constexpr Type type = t_relu;;
		};
		struct lrelu {
			static float fx(float x) { return max(0.01f * x, x); }
			static float dx(float x) { return (x > 0.f) * 0.99f + 0.01f; }
			static constexpr Type type = t_lrelu;
		};
		struct sat {
			static float fx(float x) { return min(max(x, 0.f), 1.f); }
			static float dx(float x) { return (x > 0.f) * (x < 1.f); }
			static constexpr Type type = t_sat;
		};
		struct sig {
			static float fx(float x) { return 1.f / (1.f + expf(-x)); }
			static float dx(float x) { return fx(x) * (1.f - fx(x)); }
			static constexpr Type type = t_sig;
		};
		struct clu {
			static float fx(float x) { return min(max(x, -1.f), 1.f); }
			static float dx(float x) { return (x > -1.f) * (x < 1.f); }
			static constexpr Type type = t_clu;
		};
		struct swish {
			static float fx(const float x) { return x * sig::fx(x); }
			static float dx(const float x) { return x * sig::dx(x) + sig::fx(x); }
			static constexpr Type type = t_swish;
		};
		struct tanh {
			static float fx(float x) { return 2.f / (1.f + expf(-2.f * x)) - 1.f; }
			static float dx(float x) { return 1.f - fx(x) * fx(x); }
			static constexpr Type type = t_tanh;
		};
	};
}