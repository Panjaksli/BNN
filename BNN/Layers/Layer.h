#pragma once
#include "../Eigen_util.h"
#include "../Afun.h"
namespace BNN {
	class Layer {
	public:
		Layer() {}
		Layer(dim1<3> dims, Layer* _prev = nullptr) : a(dims), prev(_prev) { if (prev)prev->set_next(this); }
		virtual ~Layer() {}
		//init weights, biases etc...
		virtual void init() {}
		//update the filter -> currently just dropout layer
		virtual void update() {}
		//thread safe but slower
		virtual Tensor compute(const Tensor& x) const = 0;
		//not thread safe but faster
		virtual const Tensor& predict() = 0;
		//propagates gradient backwards
		virtual void derivative() {}
		//propagates gradient and accumulates weights/filter
		virtual void gradient(Tensor& dw, Tensor& db, float inv_n = 1.f) {}
		//output of prev layer -> x and current layer -> y
		inline Tensor& x() { return prev->a; }
		inline Tensor& y() { return a; }
		inline const Tensor& x()const { return prev->a; }
		inline const Tensor& y()const { return a; }
		//get weights and biases if they exist
		virtual Tensor* get_w() { return nullptr; }
		virtual Tensor* get_b() { return nullptr; }
		//dims of prev layer öutput -> x and current layer output -> y
		idx sz_x() const { return prev->a.size(); }
		dim1<3> dim_x() const { return prev->a.dimensions(); }
		idx dim_x(idx i) const { return prev->a.dimension(i); }
		idx sz_y() const { return a.size(); }
		dim1<3> dim_y() const { return a.dimensions(); }
		idx dim_y(idx i) const { return a.dimension(i); }
		
		//real dims of layer inputs / outputs
		virtual dim1<3> dim_in() const { return a.dimensions(); }
		virtual idx sz_in() const { return a.size(); }
		dim1<3> dim_out() const { return dim_y(); }
		idx sz_out() const { return sz_y(); }
		//dims of w and b if applicable
		virtual dim1<3> dim_w() const { return { 0,0,0 }; }
		virtual dim1<3> dim_b() const { return { 0,0,0 }; }
		virtual idx dim_w(idx i) const { return 0; }
		virtual idx dim_b(idx i) const { return 0; }
		virtual float get_cost() const { return next->get_cost(); }
		virtual void save(std::ostream& os) const {}
		virtual void load(std::istream& is) const {}
		virtual void print() const {}
		virtual bool compile(Layer* pnode, Layer* nnode) {
			set_prev(pnode);
			set_next(nnode);
			return in_eq_out();
		}
		Layer* first() {
			if (prev) return prev->first();
			else return this;
		}
		const Layer* first()const {
			if (prev) return prev->first();
			else return this;
		}
		const Layer* last()const {
			if (next) return next->last();
			else return this;
		}
		Layer* last() {
			if (next) return next->last();
			else return this;
		}
	protected:
		void set_next(Layer* node) { next = node; }
		void set_prev(Layer* node) { prev = node; }
		bool in_eq_out() const {
			return sz_x() == sz_in();
		}
	private:
		Tensor a;
	public:
		Layer* prev = nullptr;
		Layer* next = nullptr;
	};

} // namespace BNN
