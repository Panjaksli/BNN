#pragma once
#include "../Misc/Eigen_util.h"
#include "../Misc/Afun.h"
namespace BNN {
	enum LType {
		T_None, t_Input, t_Output, t_Dense, t_Conv, t_TConv, t_AvgPool, t_AvgUpool, t_Dropout
	};
	class Layer {
	public:
		Layer() {}
		Layer(dim1<3> dims, Layer* _prev = nullptr) : a(dims), prev(_prev) { if(prev)prev->set_next(this); }
		virtual ~Layer() {}
		//random init weights, biases etc...
		virtual void init() {}
		//zero weights/biases
		void zero() { if(get_w()) get_w()->setZero(); if(get_b()) get_b()->setZero(); }
		//update the filter -> currently just dropout layer
		virtual void update() {}
		//thread safe but slower
		virtual Tensor compute(const Tensor& x) const = 0;
		//compute result agnostic to the size of underlying nodes
		virtual Tensor comp_dyn(const Tensor& x) const = 0;
		virtual float error(const Tensor& y0) { return 1e6f; }
		//not thread safe but faster
		virtual const Tensor& predict() = 0;
		virtual const Tensor& predict(const Tensor& x) { return next->predict(x); };
		//propagates gradient backwards
		virtual void derivative(bool ptrain) {}
		//propagates gradient and accumulates weights/filter
		virtual void gradient(Tensor& dw, Tensor& db, bool ptrain, float inv_n = 1.f) {}
		virtual LType type() const = 0;
		LType ptype()const { return prev ? prev->type() : T_None; }
		LType ntype()const { return next ? next->type() : T_None; }
		//output of prev layer -> x and current layer -> y
		inline Tensor& x() { return prev->a; }
		inline Tensor& y() { return a; }
		inline const Tensor& x()const { return prev->a; }
		inline const Tensor& y()const { return a; }
		//get weights and biases if they exist
		virtual Tensor* get_w() { return nullptr; }
		virtual Tensor* get_b() { return nullptr; }
		//dims of prev layer output
		dim1<3> pdims() const { return prev ? prev->odims() :dim1<3>{0, 0, 0}; }
		idx pdim(idx i) const { return pdims()[i]; }
		idx psize() const { return prev ? prev->osize() : 0; }
		//real dims of layer inputs
		virtual dim1<3> idims() const { return a.dimensions(); }
		idx isize() const { return product(idims()); }
		idx idim(idx i) const { return idims()[i]; }
		//output dims of the layer
		dim1<3> odims() const { return a.dimensions(); }
		idx odim(idx i) const { return odims()[i]; }
		idx osize()const { return product(odims()); }
		//dims of w and b if applicable
		virtual dim1<3> wdims() const { return { 0,0,0 }; }
		virtual dim1<3> bdims() const { return { 0,0,0 }; }
		idx wdim(idx i) const { return wdims()[i]; }
		idx bdim(idx i) const { return bdims()[i]; }
		bool trainable() const { return type() == t_Conv || type() == t_TConv || type() == t_Dense;}
		virtual void save(std::ostream& os) const {}
		virtual void print() const {}
		bool compile(Layer* pnode, Layer* nnode) {
			set_prev(pnode);
			set_next(nnode);
			return !pnode || in_eq_out();
		}
		Layer* first() {
			if(prev) return prev->first();
			else return this;
		}
		const Layer* first()const {
			if(prev) return prev->first();
			else return this;
		}
		const Layer* last()const {
			if(next) return next->last();
			else return this;
		}
		Layer* last() {
			if(next) return next->last();
			else return this;
		}
		virtual Layer* clone() const = 0;
	protected:
		void set_next(Layer* node) { next = node; }
		void set_prev(Layer* node) { prev = node; }
		bool in_eq_out() const {
			return psize() == isize();
		}
	private:
		Tensor a;
	public:
		Layer* prev = nullptr;
		Layer* next = nullptr;
	};

} // namespace BNN
