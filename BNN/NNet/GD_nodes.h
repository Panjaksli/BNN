#pragma once
#include "../Layers/Layer.h"
namespace BNN {
	namespace GD {
		class SGD_node {
		public:
			SGD_node() : node(nullptr), valid(0) {}
			SGD_node(Layer* node) : dw(node->wdims()), db(node->bdims()), node(node),
				valid(node->trainable()) {
				init();
			}
			void get_grad(bool ptrain) {
				if(valid)node->gradient(dw, db, ptrain);
				else node->derivative(ptrain);
			}
			void update_grad(float alpha, float inv_n, Regular reg) {
				if(valid) {
					gradient(dw, *node->get_w(), inv_n,  reg);
					gradient(db, *node->get_b(), inv_n, Regular());
					update(*node->get_w(), dw, alpha);
					update(*node->get_b(), db, alpha);
					reset_grad();
				}
				node->update();
			}
			void reset_grad() {
				if(valid) {
					dw.setZero();
					db.setZero();
				}
			}
			void reset_cache() {
				if(valid)init();
			}
			void init() {
				if(valid) {
					dw.setZero();
					db.setZero();
				}
			}
			Tensor* get_vw() { return nullptr; }
			Tensor* get_vb() { return nullptr; }
			Tensor* get_mw() { return nullptr; }
			Tensor* get_mb() { return nullptr; }
			Tensor* get_db() { return &dw; }
			Tensor* get_dw() { return &db; }
			static void gradient(Tensor& dx, const Tensor& x, float inv_n,  Regular reg) {
				switch(reg.tag) {
					case RegulTag::L1: dx = dx * inv_n + reg.l1 * x.sign(); break;
					case RegulTag::L2: dx = dx * inv_n + 2.f * reg.l2 * x; break;
					case RegulTag::L1L2: dx = dx * inv_n + reg.l1 * x.sign() + 2.f * reg.l2 * x; break;
					default: dx = dx * inv_n; break;
				}
			}
		protected:
			Tensor dw, db;
			Layer* node;
			bool valid;
		private:
			static void update(Tensor& x, const Tensor& d, float alpha) {
				x = x - alpha * d;
			}
		};
		class AGD_node : public SGD_node {
		public:
			AGD_node() {}
			AGD_node(Layer* node) : SGD_node(node), vw(node->wdims()), vb(node->bdims()) { init(); }
			void update_grad(float alpha, float mu, float inv_n,  Regular reg) {
				if(valid) {
					gradient(dw, *node->get_w(), inv_n, reg);
					gradient(db, *node->get_b(), inv_n, Regular());
					update(*node->get_w(), vw, dw, alpha, mu);
					update(*node->get_b(), vb, db, alpha, mu);
					reset_grad();
				}
				node->update();
			}
			void reset_cache() {
				if(valid) init();
			}
			void init() {
				if(valid) {
					vw.setZero();
					vb.setZero();
				}
			}
			Tensor* get_vw() { return &vw; }
			Tensor* get_vb() { return &vb; }
			Tensor* get_mw() { return nullptr; }
			Tensor* get_mb() { return nullptr; }
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
			NAG_node(Layer* node) : SGD_node(node), vw(node->wdims()), vb(node->bdims()) { init(); }

			void update_grad(float alpha, float mu, float inv_n,  Regular reg) {
				if(valid) {
					gradient(dw, *node->get_w(), inv_n, reg);
					gradient(db, *node->get_b(), inv_n, Regular());
					update(*node->get_w(), vw, dw, alpha, mu);
					update(*node->get_b(), vb, db, alpha, mu);
					reset_grad();
				}
				node->update();
			}
			void reset_cache() {
				if(valid)
					init();
			}
			void init() {
				if(valid) {
					vw.setZero();
					vb.setZero();
				}
			}
			Tensor* get_vw() { return &vw; }
			Tensor* get_vb() { return &vb; }
			Tensor* get_mw() { return nullptr; }
			Tensor* get_mb() { return nullptr; }
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
			RMS_node(Layer* node) : SGD_node(node), vw(node->wdims()), vb(node->bdims()) { init(); }
			void update_grad(float alpha, float beta, float eps, float inv_n,  Regular reg) {
				if(valid) {
					gradient(dw, *node->get_w(), inv_n,  reg);
					gradient(db, *node->get_b(), inv_n, Regular());
					update(*node->get_w(), vw, dw, alpha, beta, eps);
					update(*node->get_b(), vb, db, alpha, beta, eps);
					reset_grad();
				}
				node->update();
			}
			void reset_cache() {
				if(valid) init();
			}
			void init() {
				if(valid) {
					vw.setZero();
					vb.setZero();
				}
			}
			Tensor* get_vw() { return &vw; }
			Tensor* get_vb() { return &vb; }
			Tensor* get_mw() { return nullptr; }
			Tensor* get_mb() { return nullptr; }
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
			ADAM_node(Layer* node) : SGD_node(node), mw(node->wdims()), mb(node->bdims()), vw(node->wdims()), vb(node->bdims()) { init(); }
			void update_grad(float alpha, float beta1, float beta2, float eps, float inv_n,  Regular reg) {
				if(valid) {
					gradient(dw, *node->get_w(), inv_n,  reg);
					gradient(db, *node->get_b(), inv_n, Regular());
					update(*node->get_w(), mw, vw, dw, alpha, beta1, beta2, eps);
					update(*node->get_b(), mb, vb, db, alpha, beta1, beta2, eps);
					reset_grad();
				}
				node->update();
			}
			void reset_cache() {
				if(valid)
					init();
			}
			void init() {
				if(valid) {
					vw.setZero();
					vb.setZero();
					mw.setZero();
					mb.setZero();
				}
			}
			Tensor* get_vw() { return &vw; }
			Tensor* get_vb() { return &vb; }
			Tensor* get_mw() { return &mw; }
			Tensor* get_mb() { return &mb; }
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