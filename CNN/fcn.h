#pragma once
#include"fcn_layer.h"
#include<tuple>

struct fully_connected_network {
	size_t initial, unique;
	std::vector<std::unique_ptr<fcn_layer>> layers;
	std::vector<matrix_ptr> weights, diffs;
	std::vector<gradient> grads;
	matrix_ptr return_weights;
	flat_ptr net_loss;

	fully_connected_network(size_t initial, size_t unique);

	flat_ptr forward(const flat_ptr& flat);
	flat_ptr fast_forward(const flat_ptr& flat);
	flat_ptr backprop(size_t expected, const one_dim& actual);
	flat_ptr fcn_error(const one_dim& prev_loss);

	flat_ptr error_grad(size_t expected, const one_dim& actual);
	flat_ptr soft_arg_max(const one_dim& flat);
};