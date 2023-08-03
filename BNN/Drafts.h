#pragma once
namespace BNN {
	class Layer {
	public:
		Layer() {}
		Layer(dim1<3> dims, Layer* _prev = nullptr) : a(dims), prev(_prev) { if (prev)prev->set_next(this); }
		virtual Tensor predict(const Tensor& x) const = 0;
		virtual void forward() = 0;
		virtual void gradient(Tensor& dw, Tensor& db, float inv_n = 1.f) {}
		virtual void init() {}
		virtual void rand_init() {}
		virtual void reset_grad() {}
		virtual void reset_all() {}
		dim1<3> dim_y() const { return a.dimensions(); }
		idx dim(idx i) const { return a.dimension(i); }
		idx size() const { return a.size(); }
		void set_next(Layer* node) { next = node; }
		void set_prev(Layer* node) { prev = node; }
		inline Tensor& x() { return prev->a; }
		inline Tensor& y() { return a; }
		virtual Tensor* get_w() { return nullptr; }
		virtual Tensor* get_b() { return nullptr; }
		virtual float get_cost() const { return next->get_cost(); }
		virtual void save(std::ostream& os) const {}
		virtual void load(std::istream& is) const {}
		virtual ~Layer() {}
	private:
		Tensor a;
	protected:
		Layer* prev = nullptr;
		Layer* next = nullptr;
	};
	//Allows reshaping of the input
	class Input : public Layer {
	public:
		Input(shp3 d) : Layer(d) {}
		void input(const Tensor& x) { y() = x.reshape(dim_y()); forward(); }
		Tensor predict(const Tensor& x) const override {
			return next->predict(x.reshape(dim_y()));
		}
		Tenarr predict_batch(const Tenarr& x, dim1<3> d) const {
			Tenarr y(x.dimension(0), d[0], d[1], d[2]);
			for (int i = 0; i < x.dimension(0); i++) {
				y.chip(i, 0) = predict(x.chip(i, 0));
			}
			return y;
		}
	private:
		void forward() override { next->forward(); }
	};
	//Computes error and allows reshaping of the output
	class Output : public Layer {
	public:
		Output(shp3 d, Afun af = Afun::t_lin, Efun er = t_mse) : Output(d, nullptr, af, er) {}
		Output(shp3 d, Layer* prev, Afun af = Afun::t_lin, Efun er = t_mse) : Layer(d, prev), err(er), af(af) {}
		Tensor predict(const Tensor& x) const override {
			return x.reshape(dim_y()).unaryExpr(af.fx());
		}
		Tensor& output() { return y(); }
		float error(const Tensor& y0) {
			float cost = 0;
			if (err == t_mse) {
				y() = y() - y0;
				cost = fsca(y().square().mean()).operator()(0);
				y() = 2.f * y();
			}
			else if (err == t_mae) {
				y() = y() - y0;
				cost = fsca(y().abs().mean()).operator()(0);
				y() = y().sign();
			}
			x() = y().reshape(x().dimensions());
			return cost;
		}
	private:
		void forward() override { y() = x().reshape(dim_y()); }
		Efun err;
		Afun af;
	};
	//Dense layer, automatic reshaping of inputs/outputs
	class Dense : public Layer {
	public:
		Dense(shp2 d, Afun af = Afun::t_lrelu) : Dense(d, nullptr, af) {}
		Dense(shp2 d, Layer* prev, Afun af = Afun::t_lrelu) :
			Layer({ 1,d[0],1 }, prev), dz(dim_y()), b(dim_y()),
			w(1, d[1], prev ? prev->size() : d[0]), af(af) {
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
			return next->predict(fma(w, x.reshape(dim1<3>{ 1, w.dimension(2), 1 }), b).unaryExpr(af.fx()));
		}
		void gradient(Tensor& dw, Tensor& db, float inv_n = 1.f) override {
			dz = y() * dz;
			db += dz * inv_n;
			dw += mul(dz, x().reshape(dim1<3>{ 1, 1, w.dimension(2) }), { 1,0 })* inv_n;
			mul_r(x(), w, dz, dim1<3>{ 1, w.dimension(2), 1 }, { 0,0 });
		}
		inline Tensor* get_w() override { return &w; }
		inline Tensor* get_b() override { return &b; }
	private:
		void forward() override {
			fma_r(y(), w, x().reshape(dim1<3>{ 1, w.dimension(2), 1 }), b);
			dz = y().unaryExpr(af.dx());
			y() = y().unaryExpr(af.fx());
			next->forward();
		}
		Tensor dz, b;
		Tensor w;
		Afun af;
	};
	class Convol : public Layer {
	public:
		Convol(shp3 d, idx nch, shp2 ks, shp2 st, shp2 pa, Afun af = Afun::t_lrelu) : Convol(d, nch, ks, st, pa, nullptr, af) {}
		Convol(shp3 d, idx nch, shp2 ks, shp2 st, shp2 pa, Layer* prev, Afun af = Afun::t_lrelu) :
			Layer({ nch,c_dim(d[1],ks[0],st[0],pa[0]),c_dim(d[2],ks[1],st[1],pa[1]) }, prev),
			dz(dim_y()), b(dim_y()),
			w(d[0] * nch, ks[0], ks[1]), d_in(d),
			st(st), pa(pa), af(af) {
			_init();
		}
		void init() override {
			_init();
		}
		Tensor predict(const Tensor& x) const override {
			return next->predict((conv(x.reshape(d_in), w, st, pa) + b).unaryExpr(af.fx()));
		}
		void forward() override {
			conv_r(y(), x().reshape(d_in), w, st, pa);
			y() = y() + b;
			dz = y().unaryExpr(af.dx());
			y() = y().unaryExpr(af.fx());
			next->forward();
		}
		void gradient(Tensor& dw, Tensor& db, float inv_n = 1.f) override {
			dz = y() * dz;
			db += dz * inv_n;
			dw += iconv(x().reshape(d_in), dz, st, pa) * inv_n;
			idx p1 = c_pad(d_in[1], dz.dimension(1), w.dimension(1), st.first);
			idx p2 = c_pad(d_in[2], dz.dimension(2), w.dimension(2), st.second);;
			auto dy = dz.inflate(dim1<3>{ 1, st.first, st.second });
			conv_r(x(), dy, w.reverse(dimx<bool, 3>{false, true, true}), d_in, { 1, 1 }, { p1, p2 });
		}
	private:
		void _init() {
			b = b.setRandom() * 0.5f - 0.25f;
			w = w.setRandom() * 0.5f - 0.25f;
		}
		Tensor dz, b;
		Tensor w;
		dim1<3> d_in;
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