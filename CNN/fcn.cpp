#include<algorithm>
#include<cmath>
#include"fcn.h"

fully_connected_network::fully_connected_network(size_t initial, size_t unique) :
	initial(initial), unique(unique), net_loss(flat_share(initial)) {}

flat_ptr fully_connected_network::forward(const flat_ptr& flat) {
	flat_ptr vectorised = flat_ptr(flat);
	for (auto& layer : layers) { vectorised = layer->forward(vectorised); }
	return soft_arg_max(*vectorised);
}

flat_ptr fully_connected_network::fast_forward(const flat_ptr& flat) {
	flat_ptr vectorised = flat_ptr(flat);
	for (auto& layer : layers) { vectorised = layer->fast_forward(*vectorised); }
	return soft_arg_max(*vectorised);
}

flat_ptr fully_connected_network::backprop(size_t expected, const one_dim& actual) {
	flat_ptr loss = error_grad(expected, actual);
	for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer) { loss = (*layer)->backprop(*loss); }
	for (size_t w = 0; w < weights.size(); ++w) { grads[w].descent(weights[w]->kern, diffs[w]->kern); }
	return fcn_error(*loss);
}

flat_ptr fully_connected_network::fcn_error(const one_dim& prev_loss) {
	reset_to_zero(*net_loss);
	for (size_t in = 0; in < initial; ++in) {
		for (size_t prev = 0; prev < return_weights->size.second; ++prev) {
			net_loss->at(in) = prev_loss[prev] * return_weights->at(in, prev);
		}
	}
	return net_loss;
}

flat_ptr fully_connected_network::soft_arg_max(const one_dim& flat) {
	float maximal = *std::max_element(flat.begin(), flat.end());
	flat_ptr softed = flat_share(flat.size());
	float normaliser = 0.0f;
	for (size_t el = 0; el < flat.size(); ++el) {
		softed->at(el) = std::exp(flat[el] - maximal);
		normaliser += softed->at(el);
	}
	for (float& element : *softed) { element /= normaliser; }
	return softed;
}

flat_ptr fully_connected_network::error_grad(size_t expected, const one_dim& actual) {
	one_dim grad = actual;
	grad[expected] -= 1.0f;
	return std::make_shared<one_dim>(std::move(grad));
}