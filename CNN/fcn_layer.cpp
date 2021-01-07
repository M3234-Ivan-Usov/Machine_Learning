#include"fcn_layer.h"

fcn_layer::fcn_layer(size_t neurons, matrix_ptr& weights_in, matrix_ptr& weights_in_diffs) :
	neurons(neurons), weights_in(weights_in), weights_in_diffs(weights_in_diffs),
	activator(std::make_unique<identity_activator<one_dim>>()),
	layer_loss(std::make_shared<one_dim>(neurons)),
	layer_value(neurons) {}

flat_ptr fcn_layer::forward(const flat_ptr& prev_layer) {
	last_input = flat_ptr(prev_layer);
	reset_to_zero(layer_value);
	for (size_t in = 0; in < weights_in->size.first; ++in) {
		for (size_t to = 0; to < weights_in->size.second; ++to) {
			layer_value[to] += prev_layer->at(in) * weights_in->at(in, to);
		}
	}
	last_grad = activator->gradiate(layer_value);
	return activator->activate(layer_value);
}

flat_ptr fcn_layer::fast_forward(const one_dim& prev_layer) {
	reset_to_zero(layer_value);
	for (size_t in = 0; in < weights_in->size.first; ++in) {
		for (size_t to = 0; to < weights_in->size.second; ++to) {
			layer_value[to] += prev_layer[in] * weights_in->at(in, to);
		}
	}
	return activator->activate(layer_value);
}


fcn_hidden::fcn_hidden(size_t neurons, matrix_ptr weights_in, matrix_ptr weights_in_diffs) :
	fcn_layer(neurons, weights_in, weights_in_diffs) {}

flat_ptr fcn_hidden::backprop(const one_dim& post_layer_loss) {
	reset_to_zero(*layer_loss), reset_to_zero(weights_in_diffs->kern);
	for (size_t to = 0; to < weights_out->size.first; ++to) {
		for (size_t from = 0; from < weights_out->size.second; ++from) {
			layer_loss->at(to) += post_layer_loss[from] * weights_out->at(to, from);
		}
		layer_loss->at(to) *= last_grad->at(to);
		for (size_t from = 0; from < weights_in->size.first; ++from) {
			weights_in_diffs->at(from, to) = layer_loss->at(to) * last_input->at(from);
		}
	}
	last_grad.reset(), last_input.reset();
	return layer_loss;
}


fcn_output::fcn_output(size_t neurons, matrix_ptr weights_in, matrix_ptr weights_in_diffs) :
	fcn_layer(neurons, weights_in, weights_in_diffs) {}

flat_ptr fcn_output::backprop(const one_dim& post_layer_loss) {
	reset_to_zero(*layer_loss), reset_to_zero(weights_in_diffs->kern);
	for (size_t out = 0; out < neurons; ++out) {
		layer_loss->at(out) = last_grad->at(out) * post_layer_loss[out];
		for (size_t from = 0; from < weights_in->size.first; ++from) {
			weights_in_diffs->at(from, out) = layer_loss->at(out) * last_input->at(from);
		}
	}
	last_input.reset(), last_grad.reset();
	return layer_loss;
}
