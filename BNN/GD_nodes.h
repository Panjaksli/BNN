#pragma once
#include "Layer.h"
namespace BNN {
	namespace GD {
		class SGD_node {
		public:
			SGD_node() : layer(nullptr), valid(0) {}
			SGD_node(Layer* layer) : dw(layer->dim_w()), db(layer->dim_b()), layer(layer), valid(1) { init(); }
			void get_grad(float inv_n) {
				if (valid)layer->gradient(dw, db, inv_n);
				else layer->derivative();
			}
			void update_grad(float alpha) {
				if (valid) {
					update(*layer->get_w(), dw, alpha);
					update(*layer->get_b(), db, alpha);
					reset_grad();
				}
			}
			void reset_grad() {
				if (valid)init();
			}
			void reset_cache() {
				if (valid)init();
			}
			void init() {
				if (valid) {
					dw.setZero();
					db.setZero();
				}
			}
		protected:
			Tensor dw, db;
			Layer* layer;
			bool valid;
		private:
			static void update(Tensor& x, const Tensor& d, float alpha) {
				x = x - alpha * d;
			}
		};
		class AGD_node : public SGD_node {
		public:
			AGD_node() {}
			AGD_node(Layer* layer) : SGD_node(layer), vw(layer->dim_w()), vb(layer->dim_b()) { init(); }
			void update_grad(float alpha, float mu) {
				if (valid) {
					update(*layer->get_w(), vw, dw, alpha, mu);
					update(*layer->get_b(), vb, db, alpha, mu);
					reset_grad();
				}
			}
			void reset_cache() {
				if (valid) init();
			}
			void init() {
				if (valid) {
					vw.setZero();
					vb.setZero();
				}
			}
		protected:
			Tensor vw, vb;
		private:
			static void update(Tensor& x, Tensor& v, const Tensor& d, float alpha, float mu) {
				v = mu * v + (1.f - mu) * d;
				x = x - alpha * v;
			}
		};
		class NAG_node : public SGD_node {
		public:
			NAG_node() {}
			NAG_node(Layer* layer) : SGD_node(layer), vw(layer->dim_w()), vb(layer->dim_b()) { init(); }

			void update_grad(float alpha, float mu) {
				if (valid) {
					update(*layer->get_w(), vw, dw, alpha, mu);
					update(*layer->get_b(), vb, db, alpha, mu);
					reset_grad();
				}
			}
			void reset_cache() {
				if (valid)
					init();
			}
			void init() {
				if (valid) {
					vw.setZero();
					vb.setZero();
				}
			}
		protected:
			Tensor vw, vb;
		private:
			static void update(Tensor& x, Tensor& v, const Tensor& d, float alpha, float mu) {
				x = x + mu * mu * v - alpha * d;
				v = mu * v - alpha * d;
			}
		};
		class RMS_node : public SGD_node {
		public:
			RMS_node() {}
			RMS_node(Layer* layer) : SGD_node(layer), vw(layer->dim_w()), vb(layer->dim_b()) { init(); }
			void update_grad(float alpha, float beta, float eps) {
				if (valid) {
					update(*layer->get_w(), vw, dw, alpha, beta, eps);
					update(*layer->get_b(), vb, db, alpha, beta, eps);
					reset_grad();
				}
			}
			void reset_cache() {
				if (valid) init();
			}
			void init() {
				if (valid) {
					vw.setZero();
					vb.setZero();
				}
			}
		protected:
			Tensor vw, vb;
		private:
			static void update(Tensor& x, Tensor& v, const Tensor& d, float alpha, float beta, float eps) {
				v = beta * v + (1.f - beta) * d.square();
				x = x - alpha * d * (v + eps).rsqrt();
			}
		};
		class ADAM_node : public SGD_node {
		public:
			ADAM_node() {}
			ADAM_node(Layer* layer) : SGD_node(layer), mw(layer->dim_w()), mb(layer->dim_b()), vw(layer->dim_w()), vb(layer->dim_b()) { init(); }
			void update_grad(float alpha, float beta1, float beta2, float eps) {
				if (valid) {
					update(*layer->get_w(), mw, vw, dw, alpha, beta1, beta2, eps);
					update(*layer->get_b(), mb, vb, db, alpha, beta1, beta2, eps);
					reset_grad();
				}
			}
			void reset_cache() {
				if (valid)
					init();
			}
			void init() {
				if (valid) {
					vw.setZero();
					vb.setZero();
					mw.setZero();
					mb.setZero();
				}
			}
		protected:
			Tensor mw, mb;
			Tensor vw, vb;
		private:
			static void update(Tensor& x, Tensor& m, Tensor& v, const Tensor& d, float alpha, float beta1, float beta2, float eps) {
				m = beta1 * m + (1.f - beta1) * d;
				v = beta2 * v + (1.f - beta2) * d.square();
				auto mt = m * (1.f / (1.f - beta1 * beta1));
				auto vt = v * (1.f / (1.f - beta2 * beta2));
				x = x - alpha * mt * (vt + eps).rsqrt();
			}
		};
	}
}