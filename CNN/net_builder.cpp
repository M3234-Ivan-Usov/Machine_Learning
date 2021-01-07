#include"net_builder.h"
#include<stdexcept>
#include<sstream>
#include<iostream>

net_builder::cnn net_builder::request(size_t im_height, 
	size_t im_width, size_t labels) { return cnn(im_height, im_width, labels); }

net_builder::cnn::cnn(size_t height, size_t width, size_t labels) :
	net(std::make_shared<convolutional_network>(height, width, labels)) {
	std::ostringstream layer;
	layer << "Input : " << 1 << " x " << height << " x " << width;
	net_view.push_back(layer.str());
}

net_builder::cnn& net_builder::cnn::conv(size_t kernels, size_t y, size_t x) {
	convolution_layer worker(kernels, { y, x }, net->depth);
	dim_2_sizes out_size = worker.prepare(net->height, net->width);
	net->layers.push_back(std::make_unique<convolution_layer>(std::move(worker)));
	net->depth *= kernels;
	net->height = out_size.first;
	net->width = out_size.second;
	std::ostringstream layer;
	layer << "Conv " << kernels << " x " << y << " x " << x << " : " << 
		net->depth << " x " << net->height << " x " << net->width;
	net_view.push_back(layer.str());
	return *this;
}

net_builder::cnn& net_builder::cnn::pool(size_t stride) {
	if (net->width < stride || net->height < stride) { 
		throw std::runtime_error("Attempt to pool too small matrix"); 
	}
	pooling_layer worker(stride);
	auto out_size = worker.prepare(net->height, net->width, net->depth);
	net->layers.push_back(std::make_unique<pooling_layer>(std::move(worker)));
	net->height = out_size.first;
	net->width = out_size.second;
	std::ostringstream layer;
	layer << "Pool " << stride << " x " << stride << " : " << 
		net->depth << " x " << net->height << " x " << net->width;
	net_view.push_back(layer.str());
	return *this;
}

net_builder::cnn& net_builder::cnn::relu(float alpha) {
	auto worker = std::make_unique<relu_activator<two_dim>>(alpha);
	net->layers.push_back(std::make_unique<cnn_activator>(std::move(worker), net->depth));
	return *this;
}

net_builder::cnn& net_builder::cnn::sigmoid() {
	auto worker = std::make_unique<sigmoid_activator<two_dim>>();
	net->layers.push_back(std::make_unique<cnn_activator>(std::move(worker), net->depth));
	return *this;
}

net_builder::cnn& net_builder::cnn::softplus() {
	auto worker = std::make_unique<softplus_activator<two_dim>>();
	net->layers.push_back(std::make_unique<cnn_activator>(std::move(worker), net->depth));
	return *this;
}

net_builder::cnn& net_builder::cnn::tanh() {
	auto worker = std::make_unique<tanh_activator<two_dim>>();
	net->layers.push_back(std::make_unique<cnn_activator>(std::move(worker), net->depth));
	return *this;
}

net_builder::fcn net_builder::cnn::full_con() { 
	net->out_tensor_size = net->depth * net->height * net->width;
	return fcn(net, net_view);
}

net_builder::fcn::fcn(shared_cnn owner, view net_view) : owner(owner), net_view(net_view) {
	net = std::make_shared<fully_connected_network>(owner->out_tensor_size, owner->unique);
}

net_builder::fcn& net_builder::fcn::hidden(size_t neurons) {
	if (owner->unique > neurons) { 
		throw std::logic_error("Attempt to create less neurons than labels"); 
	}
	size_t in_size = (net->layers.empty()) ? net->initial : net->layers.back()->neurons;
	if (in_size < neurons) { 
		throw std::logic_error("Attempt to create more neurons that on previous layer");
	}
	net->weights.emplace_back(matrix::generate({ in_size, neurons }));
	net->diffs.emplace_back(std::make_shared<matrix>(std::make_pair(in_size, neurons)));
	net->grads.emplace_back(in_size * neurons);
	if (!net->layers.empty()) { dynamic_cast<fcn_hidden&>(*net->layers.back()).weights_out = net->weights.back(); }
	net->layers.emplace_back(std::make_unique<fcn_hidden>(neurons, net->weights.back(), net->diffs.back()));
	std::ostringstream layer;
	layer << "Full: " << in_size << " ==> " << neurons;
	net_view.push_back(layer.str());
	return *this;
}

net_builder::fcn& net_builder::fcn::with_relu(float alpha) {
	net->layers.back()->activator = std::make_unique<relu_activator<one_dim>>(alpha);
	return *this;
}

net_builder::fcn& net_builder::fcn::with_tanh() {
	net->layers.back()->activator = std::make_unique<tanh_activator<one_dim>>();
	return *this;
}

net_builder::fcn& net_builder::fcn::with_sigmoid() {
	net->layers.back()->activator = std::make_unique<sigmoid_activator<one_dim>>();
	return *this;
}

net_builder::fcn& net_builder::fcn::with_softplus() {
	net->layers.back()->activator = std::make_unique<softplus_activator<one_dim>>();
	return *this;
}

net_builder::fcn& net_builder::fcn::last() {
	size_t in_size = net->layers.empty() ? net->initial : net->layers.back()->neurons;
	net->weights.emplace_back(matrix::generate({ in_size, net->unique }));
	net->diffs.emplace_back(std::make_shared<matrix>(std::make_pair(in_size, net->unique)));
	net->grads.emplace_back(in_size * net->unique);
	if (!net->layers.empty()) { dynamic_cast<fcn_hidden&>(*net->layers.back()).weights_out = net->weights.back(); }
	net->layers.emplace_back(std::make_unique<fcn_output>(net->unique, net->weights.back(), net->diffs.back()));
	std::ostringstream layer;
	layer << "Full: " << in_size << " ==> " << net->unique;
	net_view.push_back(layer.str());
	return *this;
}

convolutional_network&& net_builder::fcn::build() { 
	net->return_weights = net->weights[0];
	owner->fcn = std::shared_ptr<fully_connected_network>(net);
	std::cout << std::endl;
	for (auto layer : net_view) { std::cout << "### " << layer << std::endl; }
	std::cout << std::endl;
	return std::move(*owner); 
}

