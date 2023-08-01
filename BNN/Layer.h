#pragma once
#include "Eigen_util.h"
#include "Afun.h"

namespace BNN {
	class Layer {
	public:
		Layer() {}
		Layer(dim1<3> dims, Layer* _prev = nullptr) : a(dims), prev(_prev) { if (prev)prev->set_next(this); }
		virtual Tensor predict(const Tensor& x) const = 0;
		virtual Tenarr predict_batch(const Tenarr& x, idx d1, idx d2, idx d3) const { return Tenarr(); }
		virtual void forward(const Tensor& x) = 0;
		virtual void backward(const Tensor& dy) = 0;
		virtual void init() {}
		virtual void rand_init() {}
		virtual void reset_weig() {}
		virtual void reset_grad() {}
		virtual void reset_all() {}
		dim1<3> dim() const { return a.dimensions(); }
		idx dim(idx i) const { return a.dimension(i); }
		idx size() const { return a.size(); }
		void set_next(Layer* node) { next = node; }
		void set_prev(Layer* node) { prev = node; }
		virtual float get_cost() const { return next->get_cost(); }
		virtual void save(std::ostream& os) const {}
		virtual void load(std::istream& is) const {}
		virtual ~Layer() {}
		Tensor a;
		Layer* prev = nullptr;
		Layer* next = nullptr;
		void free() {
			if (next) next->free();
			next = prev = nullptr;
			delete this;
		}
	};
	//Allows reshaping of the input
	class Input : public Layer {
	public:
		~Input() { this->free(); }
		Input(idx d1, idx d2, idx d3) : Layer({ d1,d2,d3 }) {}
		Input(idx d1, idx d2) : Layer({ 1, d1,d2 }) {}
		Input(idx d1) : Layer({ 1,d1,1 }) {}
		Input(dim1<3> d) : Layer(d) {}
		void init() override {}
		Tensor predict(const Tensor& x) const override {
			return next->predict(x.reshape(dim()));
		}
		Tenarr predict_batch(const Tenarr& x, idx d1, idx d2, idx d3) const override {
			Tenarr y(x.dimension(0), d1, d2, d2);
			for (int i = 0; i < x.dimension(0); i++) {
				y.chip(i, 0) = predict(x.chip(i, 0));
			}
			return y;
		}
		void forward(const Tensor& x) override {
			a = x.reshape(dim());
			next->forward(a);
		}
		void backward(const Tensor& dy) override {}
	};
	//Computes error and allows reshaping of the output
	class Output : public Layer {
	public:
		Output(idx d1, idx d2, idx d3, Layer* prev, Afun af = Afun::t_lin, Efun er = t_mse) : Layer({ d1,d2,d3 }, prev), err(er), af(af) {}
		Output(idx d1, idx d2, Layer* prev, Afun af = Afun::t_lin, Efun er = t_mse) : Layer({ 1, d1,d2 }, prev), err(er), af(af) {}
		Output(idx d1, Layer* prev, Afun af = Afun::t_lin, Efun er = t_mse) : Layer({ 1,d1,1 }, prev), err(er), af(af) {}
		Output(dim1<3> d, Layer* prev, Afun af = Afun::t_lin, Efun er = t_mse) : Layer(d, prev), err(er), af(af) {}
		Tensor predict(const Tensor& x) const override {
			return x.reshape(dim()).unaryExpr(af.fx());
		}
		void forward(const Tensor& x) override {
			a = x.reshape(dim());
		}
		void backward(const Tensor& y) override {
			if (err == t_mse) {
				a = a - y;
				cost = fsca(a.square().sum()).operator()(0);
				prev->backward(2.f * a);
			}
			else if (err == t_mae) {
				a = a - y;
				cost = fsca(a.abs().sum()).operator()(0);
				prev->backward(a.sign());
			}
		}
		float get_cost() const override { return cost; }
		float cost = 1e6f;
		Efun err;
		Afun af;
	};
	//Dense layer, automatic reshaping of inputs/outputs
	class Dense : public Layer {
	public:
		Dense(idx d, Layer* prev, Afun af = Afun::t_lrelu) :
			Layer({ 1,d,1 }, prev), dz(dim()), b(dim()),
			w(1, d, prev->size()), af(af) {
			prev->set_next(this);
			_init();
		}
		void _init() {
			b = b.setRandom() * 0.5f - 0.25f;
			w = w.setRandom() * 0.5f - 0.25f;
		}
		void init() override {
			_init();
		}
		Tensor predict(const Tensor& x) const override {
			return next->predict(fma(w, x.reshape(dim1<3>{1, w.dimension(2), 1}), b).unaryExpr(af.fx()));
		}
		void forward(const Tensor& x) override {
			fma_e(a, w, x.reshape(dim1<3>{1, w.dimension(2), 1}), b);
			dz = a.unaryExpr(af.dx());
			a = a.unaryExpr(af.fx());
			next->forward(a);
		}
		void backward(const Tensor& y) override {
			dz = y.reshape(dz.dimensions()) * dz;
			auto x = prev->a.reshape(dim1<3>{1, 1, w.dimension(2)});
			Tensor dw = mul(dz, x, { 1,0 });
			auto dx = mul(w, dz, { 0,0 });
			w -= dw * 0.001f;
			b -= dz * 0.001f;
			prev->backward(dx);
		}
		Tensor dz, b;
		Tensor w;
		Afun af;
	};
	class Convol : public Layer {
	public:
		Convol(dim1<3> d, dim1<2> ks, idx st, idx pa, Layer* prev, Afun af = Afun::t_lrelu) :
			Layer({ ks[0],c_dim(d[1],ks[1],st,pa),c_dim(d[2],ks[1],st,pa) }, prev),
			dz(dim()), b(dim()),
			w(d[0] * ks[0], ks[1], ks[1]), din(d),
			st(st, st), pa(pa, pa), af(af) {
			prev->set_next(this);
			_init();
		}
		Convol(dim1<3> d, dim1<3> ks, dim1<2> st, dim1<2> pa, Layer* prev, Afun af = Afun::t_lrelu) :
			Layer({ ks[0],c_dim(d[1],ks[1],st[0],pa[0]),c_dim(d[2],ks[2],st[1],pa[1]) }, prev),
			dz(dim()), b(dim()),
			w(d[0] * ks[0], ks[1], ks[2]), din(d),
			st(st[0], st[1]), pa(pa[0], pa[1]), af(af) {
			prev->set_next(this);
			_init();
		}
		void _init() {
			b = b.setRandom() * 0.5f - 0.25f;
			w = w.setRandom() * 0.5f - 0.25f;
		}
		void init() override {
			_init();
		}
		Tensor predict(const Tensor& x) const override {
			return next->predict((conv(x.reshape(din), w, st, pa) + b).unaryExpr(af.fx()));
		}
		void forward(const Tensor& x) override {
			conv_e(a, x.reshape(din), w, st, pa);
			a = a + b;
			dz = a.unaryExpr(af.dx());
			a = a.unaryExpr(af.fx());
			next->forward(a);
		}
		void backward(const Tensor& y) override {
			dz = y.reshape(dim()) * dz;
			Tensor dw = iconv(prev->a.reshape(din), dz, st, pa);
			idx p1 = c_pad(din[1], dz.dimension(1), w.dimension(1), st.first);
			idx p2 = c_pad(din[2], dz.dimension(2), w.dimension(2), st.second);;
			auto dy = dz.inflate(dim1<3>{1, st.first, st.second});
			auto dx = conv(dy, w.reverse(dimx<bool, 3>{false, true, true}), { 1,1 }, { p1, p2 });
			w -= dw * 0.001f;
			b -= dz * 0.001f;
			prev->backward(dx);
		}
		Tensor dz, b;
		Tensor w;
		dim1<3> din;
		pair st, pa;
		Afun af;
	};
	class Maxpool : public Layer {

	};
	class Unpool : public Layer {

	};
	class Downscl : public Layer {

	};
	class Upscl : public Layer {

	};
} // namespace BNN