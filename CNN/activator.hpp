#pragma once
#include"base.h"
#include<cmath>

template <typename any_array>
struct activation {
	using any_ptr = std::shared_ptr<any_array>;
	virtual any_ptr activate(const any_array& val) {
		any_ptr out = std::make_shared<any_array>(any_array(val.size()));
		for (size_t el = 0; el < val.size(); ++el) { out->at(el) = fun(val[el]); }
		return out;
	}
	virtual any_ptr activate(const any_array& val, float bias) {
		any_ptr out = std::make_shared<any_array>(any_array(val.size()));
		for (size_t el = 0; el < val.size(); ++el) { out->at(el) = fun(val[el] + bias); }
		return out;
	}
	virtual any_ptr gradiate(const any_array& val) {
		any_ptr out = std::make_shared<any_array>(any_array(val.size()));
		for (size_t el = 0; el < val.size(); ++el) { out->at(el) = d_fun(val[el]); }
		return out;
	}
	virtual float fun(float x) = 0;
	virtual float d_fun(float x) = 0;
};

template<typename any_array>
struct identity_activator : public activation<any_array> {
	virtual float fun(float x) { return x; }
	virtual float d_fun(float x) { return 1.0f; }
};


template<typename any_array>
struct relu_activator : public activation<any_array> {
	float alpha;
	relu_activator(float alpha) : alpha(alpha) {}
	virtual float fun(float x) { return (x < 0) ? alpha * x : x; }
	virtual float d_fun(float x) { return (x < 0) ? alpha : 1.0f; }
};

template<typename any_array>
struct tanh_activator : public activation<any_array> {
	virtual float fun(float x) { return std::tanhf(x); }
	virtual float d_fun(float x) {
		float active_value = std::tanhf(x);
		return 1 - active_value * active_value;
	}
};

template<typename any_array>
struct sigmoid_activator : public activation<any_array> {
	virtual float fun(float x) { return 1.0f / (1.0f + std::exp(-x)); }
	virtual float d_fun(float x) { 
		// return activation_fun(x) * activation_fun(-x);
		float ex = std::exp(x);
		return ex / ((1.0f + ex) * (1.0f + ex));
	}
};

template<typename any_array>
struct softplus_activator : public activation<any_array> {
	virtual float fun(float x) { return std::log(1.0f + std::exp(x)); }
	virtual float d_fun(float x) { return 1.0f / (1.0f + std::exp(-x)); }
};
