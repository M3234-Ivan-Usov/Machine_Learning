#pragma once
#include"activator.hpp"
#include"grad.h"

struct fcn_layer {
	std::unique_ptr<activation<one_dim>> activator;
	flat_ptr last_input, layer_loss, last_grad;
	matrix_ptr weights_in, weights_in_diffs;
	one_dim layer_value;
	size_t neurons;

	fcn_layer(size_t neurons, matrix_ptr& weights_in, matrix_ptr& weights_in_diffs);

	virtual flat_ptr forward(const flat_ptr& flat);
	virtual flat_ptr fast_forward(const one_dim& flat);
	virtual flat_ptr backprop(const one_dim& post_layer_loss) = 0;
};

struct fcn_hidden : public fcn_layer {
	matrix_ptr weights_out;
	fcn_hidden(size_t neurons, matrix_ptr weights_in, matrix_ptr weights_in_diffs);
	virtual flat_ptr backprop(const one_dim& post_layer_loss);
};

struct fcn_output : public fcn_layer {
	fcn_output(size_t neurons, matrix_ptr weights_in, matrix_ptr weights_in_diffs);
	virtual flat_ptr backprop(const one_dim& post_layer_loss);
};
