#pragma once
#include "Eigen_util.h"
#include "Afun.h"
namespace BNN {
	class Layer {
	public:
		Layer() {}
		virtual ~Layer() {}
		Layer(dim1<3> dims, Layer* _prev = nullptr) : a(dims), prev(_prev) { if (prev)prev->set_next(this); }
		//thread safe but slower
		virtual Tensor compute(const Tensor& x) const = 0;
		//not thread safe but faster
		virtual const Tensor& predict() = 0;
		virtual void gradient(Tensor& dw, Tensor& db, float inv_n = 1.f) {}
		virtual void derivative() {}
		virtual void init() {}
		virtual void randinit() {}
		virtual void reset_grad() {}
		virtual void reset_all() {}
		void set_next(Layer* node) { next = node; }
		void set_prev(Layer* node) { prev = node; }
		inline Tensor& x() { return prev->a; }
		inline Tensor& y() { return a; }
		inline const Tensor& x()const { return prev->a; }
		inline const Tensor& y()const { return a; }
		virtual Tensor* get_w() { return nullptr; }
		virtual Tensor* get_b() { return nullptr; }
		dim1<3> dim_x() const { return prev->a.dimensions(); }
		dim1<3> dim_y() const { return a.dimensions(); }
		idx dim_x(idx i) const { return prev->a.dimension(i); }
		idx dim_y(idx i) const { return a.dimension(i); }
		idx sz_x() const { return prev->a.size(); }
		idx sz_y() const { return a.size(); }
		virtual dim1<3> dim_w() const { return { 0,0,0 }; }
		virtual dim1<3> dim_b() const { return { 0,0,0 }; }
		virtual idx dim_w(idx i) const { return 0; }
		virtual idx dim_b(idx i) const { return  0; }
		virtual float get_cost() const { return next->get_cost(); }
		virtual void save(std::ostream& os) const {}
		virtual void load(std::istream& is) const {}
		virtual void print() const {}
		virtual bool compile(Layer* pnode, Layer* nnode) { 
			set_prev(pnode); 
			set_next(nnode); 
			return in_eq_out();
		}
	protected:
		dim1<3> dim_out()const {
			if (next) return next->dim_out();
			else return dim_y();
		}
		bool in_eq_out() const {
			return prev->sz_out() == sz_in();
		}
	private:
		virtual idx sz_in() const { return a.size(); }
		virtual idx sz_out() const { return a.size(); }
		Tensor a;
	public:
		Layer* prev = nullptr;
		Layer* next = nullptr;
	};
	//Allows reshaping of the input
	class Input : public Layer {
	public:
		Input(shp3 d) : Layer(d) {}
		void input(const Tensor& x) {
			y() = x.reshape(dim_y());
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute(x.reshape(dim_y()));
		}
		Tenarr compute_batch(const Tenarr& x) const {
			dim1<3> d = dim_out();
			Tenarr y(x.dimension(0), d[0], d[1], d[2]);
			for (int i = 0; i < x.dimension(0); i++) {
				y.chip(i, 0) = compute(x.chip(i, 0));
			}
			return y;
		}
		const Tensor& predict(const Tensor& x) {
			input(x);
			return next->predict();
		}
		Tenarr predict_batch(const Tenarr& x) {
			dim1<3> d = dim_out();
			Tenarr y(x.dimension(0), d[0], d[1], d[2]);
			for (int i = 0; i < x.dimension(0); i++) {
				y.chip(i, 0) = predict(x.chip(i, 0));
			}
			return y;
		}
		bool compile(Layer* pnode, Layer* nnode) override {
			set_prev(nullptr);
			set_next(nnode);
			return true;
		}
		void print()const override {
			println("Input\t|", "\tIn:", dim_y(0), dim_y(1), dim_y(2), "\tOut:", dim_y(0), dim_y(1), dim_y(2));
		}
	private:
		const Tensor& predict() override { return next->predict(); }
	};
	//Computes error and allows reshaping of the output
	class Output : public Layer {
	public:
		Output(shp3 d, Afun af = Afun::t_lin, Efun ef = Efun::t_mse) : Output(d, nullptr, af, ef) {}
		Output(shp3 d, Layer* prev, Afun af = Afun::t_lin, Efun ef = Efun::t_mse) : Layer(d, prev), ef(ef), af(af) {}
		const Tensor& output() const { return y(); }
		float error(const Tensor& y0) {
			float cost = 0;
			if (ef.type == Efun::t_mse) {
				y() = y() - y0;
				cost = fsca(y().square().mean()).operator()(0);
				y() = 2.f * y();
			}
			else if (ef.type == Efun::t_mae) {
				y() = y() - y0;
				cost = fsca(y().abs().mean()).operator()(0);
				y() = y().sign();
			}
			x() = y().reshape(x().dimensions());
			return cost;
		}
		bool compile(Layer* pnode, Layer* nnode) override {
			set_prev(pnode);
			set_next(nullptr);
			return in_eq_out();
		}
		void print()const override {
			println("Output\t|", "\tIn:", dim_x(0), dim_x(1), dim_x(2), "\tOut:", dim_y(0), dim_y(1), dim_y(2));
		}
	private:
		Tensor compute(const Tensor& x) const override {
			return x.reshape(dim_y()).unaryExpr(af.fx());
		}
		const Tensor& predict() override { return y() = x().reshape(dim_y()).unaryExpr(af.fx()); }
		Efun ef;
		Afun af;
	};
	//Dense layer, automatic reshaping of inputs/outputs
	class Dense : public Layer {
	public:
		Dense(shp2 d, Afun af = Afun::t_swish) : Dense(d, nullptr, af) {}
		Dense(shp2 d, Layer* prev, Afun af = Afun::t_swish) :
			Layer({ 1,d[0],1 }, prev), dz(dim_y()), b(dim_y()),
			w(1, d[1], prev ? sz_x() : d[0]), af(af) {
			_init();
		}
		void init() override { _init(); }
		void derivative() override {
			dz = y() * dz;
			mul_r(x(), w, dz, dim1<3>{ 1, dim_w(2), 1 }, { 0,0 });
		}
		void gradient(Tensor& dw, Tensor& db, float inv_n = 1.f) override {
			dz = y() * dz;
			db += dz * inv_n;
			dw += mul(dz, x().reshape(dim1<3>{ 1, 1, dim_w(2) }), { 1,0 })* inv_n;
			mul_r(x(), w, dz, dim1<3>{ 1, dim_w(2), 1 }, { 0,0 });
		}
		Tensor* get_w() override { return &w; }
		Tensor* get_b() override { return &b; }
		dim1<3> dim_w() const override { return w.dimensions(); }
		dim1<3> dim_b() const override { return b.dimensions(); }
		idx dim_w(idx i) const override { return w.dimension(i); }
		idx dim_b(idx i) const override { return b.dimension(i); }
		idx sz_in() const override { return w.dimension(2); }
		idx sz_out() const override { return w.dimension(1); }
		void print()const override {
			println("Dense\t|", "\tIn:", dim_x(0), dim_x(1), dim_x(2), "\tOut:", dim_y(0), dim_y(1), dim_y(2));
		}
	private:
		void _init() {
			b = b.setRandom() * 0.5f - 0.25f;
			w = w.setRandom() * 0.5f - 0.25f;
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute(fma(w, x.reshape(dim1<3>{ 1, dim_w(2), 1 }), b).unaryExpr(af.fx()));
		}
		const Tensor& predict() override {
			fma_r(y(), w, x().reshape(dim1<3>{ 1, dim_w(2), 1 }), b);
			dz = y().unaryExpr(af.dx());
			y() = y().unaryExpr(af.fx());
			return next->predict();
		}
		Tensor dz, b;
		Tensor w;
		Afun af;
	};
	class Convol : public Layer {
	public:
		Convol(shp3 din, idx nch, shp2 ks, shp2 st, shp2 pa, Afun af = Afun::t_swish) : Convol(din, nch, ks, st, pa, nullptr, af) {}
		Convol(shp3 din, idx nch, shp2 ks, shp2 st, shp2 pa, Layer* prev, Afun af = Afun::t_swish) :
			Layer({ nch,c_dim(din[1],ks[0],st[0],pa[0]),c_dim(din[2],ks[1],st[1],pa[1]) }, prev),
			dz(dim_y()), b(dim_y()),
			w(din[0] * nch, ks[0], ks[1]), din(din),
			st(st), pa(pa), af(af) {
			_init();
		}
		Convol(idx nch, shp2 ks, shp2 st, shp2 pa, Layer* prev, Afun af = Afun::t_swish) :
			Layer({ nch,c_dim(prev->dim_y(1),ks[0],st[0],pa[0]),c_dim(prev->dim_y(2),ks[1],st[1],pa[1])}, prev),
			dz(dim_y()), b(dim_y()),
			w(prev->dim_y(0)* nch, ks[0], ks[1]), din(prev->dim_y()),
			st(st), pa(pa), af(af) {
			_init();
		}
		void init() override { _init(); }
		void derivative() override {
			dz = y() * dz;
			idx p1 = c_pad(din[1], dim_b(1), dim_w(1), st.first);
			idx p2 = c_pad(din[2], dim_b(2), dim_w(2), st.second);;
			auto dy = dz.inflate(dim1<3>{ 1, st.first, st.second });
			conv_r(x(), dy, w.reverse(dimx<bool, 3>{false, true, true}), din, { 1, 1 }, { p1, p2 });
		}
		void gradient(Tensor& dw, Tensor& db, float inv_n = 1.f) override {
			dz = y() * dz;
			db += dz * inv_n;
			dw += iconv(x().reshape(din), dz, st, pa) * inv_n;
			idx p1 = c_pad(din[1], dim_b(1), dim_w(1), st.first);
			idx p2 = c_pad(din[2], dim_b(2), dim_w(2), st.second);;
			auto dy = dz.inflate(dim1<3>{ 1, st.first, st.second });
			conv_r(x(), dy, w.reverse(dimx<bool, 3>{false, true, true}), din, { 1, 1 }, { p1, p2 });
		}
		Tensor* get_w() override { return &w; }
		Tensor* get_b() override { return &b; }
		dim1<3> dim_w() const override { return w.dimensions(); }
		dim1<3> dim_b() const override { return b.dimensions(); }
		idx dim_w(idx i) const override { return w.dimension(i); }
		idx dim_b(idx i) const override { return b.dimension(i); }
		idx sz_in() const override { return din[0] * din[1] * din[2]; }
		idx sz_out() const override { return b.size(); }
		void print()const override {
			println("Convol\t|", "\tIn:", dim_x(0), dim_x(1), dim_x(2), 
				"\tOut:", dim_y(0), dim_y(1), dim_y(2),"\tKernel:",dim_w(1), dim_w(2),"\tStride:", st[0],st[1], "\tPad:", pa[0], pa[1]);
		}
	private:
		void _init() {
			b = b.setRandom() * 0.5f - 0.25f;
			w = w.setRandom() * 0.5f - 0.25f;
		}
		Tensor compute(const Tensor& x) const override {
			return next->compute((conv(x.reshape(din), w, st, pa) + b).unaryExpr(af.fx()));
		}
		const Tensor& predict() override {
			conv_r(y(), x().reshape(din), w, st, pa);
			y() = y() + b;
			dz = y().unaryExpr(af.dx());
			y() = y().unaryExpr(af.fx());
			return next->predict();
		}
		Tensor dz, b;
		Tensor w;
		dim1<3> din;
		shp2 st, pa;
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