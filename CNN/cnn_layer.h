#pragma once
#include"activator.hpp"
#include"grad.h"

struct cnn_layer {
	dim_2_sizes in_size, out_size;
	tensor_ptr next_tensor, next_loss;

	virtual tensor_ptr forward(const tensor_ptr& input_tensor) = 0;
	virtual tensor_ptr fast_forward(const tensor& input_tensor) = 0;
	virtual tensor_ptr backprop(const tensor& tensor_diff) = 0;

	virtual matrix_ptr make_padding(const matrix& in_image, dim_2_sizes extra);
};


struct cnn_activator : public cnn_layer {
	std::unique_ptr<activation<two_dim>> worker;
	std::unique_ptr<gradient> grad;
	std::vector<float> bias, diff;
	tensor_ptr last_diff;

	cnn_activator(std::unique_ptr<activation<two_dim>> worker, size_t depth);

	virtual tensor_ptr forward(const tensor_ptr& input_tensor);
	virtual tensor_ptr fast_forward(const tensor& input_tensor);
	virtual tensor_ptr backprop(const tensor& tensor_diff);
};


struct convolution_layer : public cnn_layer {
	std::vector<matrix> kernels, flip_kernels;
	std::vector<gradient> grads;
	dim_2_sizes lin_size, extend;
	tensor_ptr last_input, diff;
	size_t out_depth;

	convolution_layer(size_t kerns, dim_2_sizes lin_size, size_t in_depth);

	virtual tensor_ptr forward(const tensor_ptr& input_tensor);
	virtual tensor_ptr fast_forward(const tensor& input_tensor);
	virtual tensor_ptr backprop(const tensor& tensor_diff);

	dim_2_sizes prepare(size_t height, size_t width);
	matrix_ptr convolve(const matrix& im, const matrix& kernel, dim_2_sizes new_size);
};


struct pooling_layer : public cnn_layer {
	using map_maximals = std::vector<dim_2_sizes>;
	size_t stride;
	dim_2_sizes miss;
	std::vector<map_maximals> maximals;

	pooling_layer(size_t stride);

	virtual tensor_ptr forward(const tensor_ptr& input_tensor);
	virtual tensor_ptr fast_forward(const tensor& input_tensor);
	virtual tensor_ptr backprop(const tensor& tensor_diff);

	dim_2_sizes prepare(size_t height, size_t width, size_t depth);
	matrix_ptr pool(const matrix& im, size_t im_index);
};
