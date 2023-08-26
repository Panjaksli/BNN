#pragma once
#include "Misc.h"
namespace BNN {
	enum Regul {L0, L1, L2};
	class Efun {
	public:
		enum Type { t_mse, t_mae } type = t_mse;
		Efun() {}
		Efun(Type type) : type(type) {}
		inline auto fx() const {
			switch(type) {
				case t_mse: return mse::fx;
				case t_mae: return mae::fx;
				default: return mse::fx;
			}
		}
		inline auto dx() const {
			switch(type) {
				case t_mse: return mse::dx;
				case t_mae: return mae::dx;
				default: return mse::dx;
			}
		}
		inline const char* name() const {
			switch(type) {
				case t_mse: return "mse";
				case t_mae: return "mae";
				default: return "null";
			}
		}
		struct mse {
			static float fx(float x, float y) { return (x - y) * (x - y); }
			static float dx(float x, float y) { return 2.f * (x - y); }
			static constexpr Type type = t_mse;
		};
		struct mae {
			static float fx(float x, float y) { return fabsf(x - y); }
			static float dx(float x, float y) { return copysignf(1.f, x - y); }
			static constexpr Type type = t_mae;
		};
	};
	class Afun {
	public:
		enum Type { t_lin, t_relu, t_lrelu, t_sat, t_sig, t_clu, t_swish, t_tanh, t_cub, t_cubl } type = t_relu;
		Afun() {}
		Afun(Type type) : type(type) {}
		inline auto fx() const {
			switch(type) {
				case t_lin: return lin::fx;
				case t_relu: return relu::fx;
				case t_lrelu: return lrelu::fx;
				case t_sat: return sat::fx;
				case t_sig: return sig::fx;
				case t_clu: return clu::fx;
				case t_swish: return swish::fx;
				case t_tanh: return tanh::fx;
				case t_cub: return cub::fx;
				case t_cubl: return cubl::fx;
				default: return lin::fx;
			}
		}
		inline auto dx() const {
			switch(type) {
				case t_lin: return lin::dx;
				case t_relu: return relu::dx;
				case t_lrelu: return lrelu::dx;
				case t_sat: return sat::dx;
				case t_sig: return sig::dx;
				case t_clu: return clu::dx;
				case t_swish: return swish::dx;
				case t_tanh: return tanh::dx;
				case t_cub: return cub::dx;
				case t_cubl: return cubl::dx;
				default: return lin::dx;
			}
		}
		inline const char* name() const {
			switch(type) {
				case t_lin: return "lin";
				case t_relu: return "relu";
				case t_lrelu: return "lrelu";
				case t_sat: return "sat";
				case t_sig: return "sig";
				case t_clu: return "clu";
				case t_swish: return "swish";
				case t_tanh: return "tanh";
				case t_cub: return "cub";
				case t_cubl: return "cubl";
				default: return "null";
			}
		}
		struct lin {
			static float fx(float x) { return x; }
			static float dx(float x) { return 1.f; }
			static constexpr Type type = t_lin;
		};
		struct relu {
			static float fx(float x) { return fmaxf(x, 0.f); }
			static float dx(float x) { return (x > 0.f); }
			static constexpr Type type = t_relu;;
		};
		struct lrelu {
			static float fx(float x) { return fmaxf(0.01f * x, x); }
			static float dx(float x) { return (x > 0.f) * 0.99f + 0.01f; }
			static constexpr Type type = t_lrelu;
		};
		struct sat {
			static float fx(float x) { return fminf(fmaxf(x, 0.f), 1.f); }
			static float dx(float x) { return (x > 0.f) * (x < 1.f); }
			static constexpr Type type = t_sat;
		};
		struct sig {
			static float fx(float x) { return 1.f / (1.f + expf(-x)); }
			static float dx(float x) { return fx(x) * (1.f - fx(x)); }
			static constexpr Type type = t_sig;
		};
		struct clu {
			static float fx(float x) { return fminf(fmaxf(x, -1.f), 1.f); }
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
		//like swish and cubl but cubic growth
		struct cub {
			static float fx(float x) {
				float x1 = fmaxf(a * x + 1.f, 0.f);
				float x2 = fmaxf(b * x + 1.f, 0.f);
				return x1 * x1 * x1 - x2 * x2;
			}
			static float dx(float x) {
				float x1 = fmaxf(a * x + 1.f, 0.f);
				float x2 = fmaxf(b * x + 1.f, 0.f);
				return (3.f * a) * x1 * x1 - (2.f * b) * x2;
			}
			static constexpr Type type = t_cub;
			static constexpr float a = 0.3;
			static constexpr float b = 0.2;
		};
		//poor mans swish (lin growth after x ~ 0.928)
		//shines best in multilayer CNNs, where relu and lrelu struggle
		//is much faster than swish (no exp, even no cmov or branching)
		struct cubl {
			static float fx(float x) {
				float x1 = fmaxf(a * x + 1.f, 0.f);
				float x2 = fmaxf(b * x + 1.f, 0.f);
				return fminf(x1 * x1 * x1 - x2 * x2, fmaxf(x - c, d));
			}
			static float dx(float x) {
				float x1 = fmaxf(a * x + 1.f, 0.f);
				float x2 = fmaxf(b * x + 1.f, 0.f);
				return fminf((3.f * a) * x1 * x1 - (2.f * b) * x2, 1.f);
			}
			static constexpr Type type = t_cubl;
			static constexpr float a = 0.3;
			static constexpr float b = 0.2;
			static constexpr float c = 0.24435912045;
			static constexpr float d = 0.68914872895;
		};
	};
}
//just storing working coeffs
// best (derivative goes through 0.5 at x = 0)
//static constexpr float a = 0.3;
//static constexpr float b = 0.2;
//static constexpr float c = 0.24435912045;
//static constexpr float d = 0.68914872895;
// worse ?
//static constexpr float a = 0.297;
//static constexpr float b = 0.213;
//static constexpr float c = 0.28981184835;
//static constexpr float d = 0.74010139985;