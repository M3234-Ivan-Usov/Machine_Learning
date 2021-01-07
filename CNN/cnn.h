#pragma once
#include"cnn_layer.h"
#include"fcn.h"
#include<memory>

using images = std::vector<matrix>;
using labels = std::vector<size_t>;

struct convolutional_network {
	using full_con = std::shared_ptr<fully_connected_network>;
	std::vector<std::unique_ptr<cnn_layer>> layers;
	size_t depth, height, width, unique;
	size_t out_tensor_size = 0;
	full_con fcn;

	convolutional_network(size_t im_height, size_t im_size, size_t labels);

	void fit(const images& train, const labels& targets);
	labels predict(const images& test);

	flat_ptr flatten(const tensor& input_tensor);
	tensor_ptr deflate(const one_dim& inflatted);
};
