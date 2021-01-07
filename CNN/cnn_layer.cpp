#include"cnn_layer.h"
#include<random>
#include<numeric>

matrix_ptr cnn_layer::make_padding(const matrix& im, dim_2_sizes extra) {
	size_t new_height = im.size.first + extra.first;
	size_t new_width = im.size.second + extra.second;
	size_t i = extra.first / 2, j = extra.second/ 2;
	matrix_ptr extended = std::make_shared<matrix>(std::make_pair(new_height, new_width));
	for (size_t y = 0; y < im.size.first; ++y) {
		for (size_t x = 0; x < im.size.second; ++x) { extended->at(y + i, x + j) = im.at(y, x); }
	}
	return extended;
}


cnn_activator::cnn_activator(std::unique_ptr<activation<two_dim>> worker, size_t depth) :
	worker(std::move(worker)), bias(std::move(*generate_flat(depth))) {
	grad = std::make_unique<gradient>(depth);
	last_diff = std::make_shared<tensor>(depth);
	next_tensor = std::make_shared<tensor>(depth);
	next_loss = std::make_shared<tensor>(depth);
	diff = std::vector<float>(depth);
}

tensor_ptr cnn_activator::forward(const tensor_ptr& prev_tensor) {
	last_diff->kern.clear(), next_tensor->kern.clear();
	size_t index = 0;
	for (const matrix& im : prev_tensor->kern) {
		last_diff->kern.emplace_back(std::move(*worker->gradiate(im.kern)), im.size);
		next_tensor->kern.emplace_back(std::move(*worker->activate(im.kern, bias[index++])), im.size);
	}
	return next_tensor;
}

tensor_ptr cnn_activator::fast_forward(const tensor& prev_tensor) {
	next_tensor->kern.clear();
	size_t index = 0;
	for (const matrix& im : prev_tensor.kern) {
		next_tensor->kern.emplace_back(std::move(*worker->activate(im.kern, bias[index++])), im.size);
	}
	return next_tensor;
}

tensor_ptr cnn_activator::backprop(const tensor& prev_loss) {
	next_loss->kern.clear(), reset_to_zero(diff);
	size_t index = 0;
	for (const matrix& im : prev_loss.kern) {
		next_loss->kern.push_back(std::move(*last_diff->kern[index].scalar_dot(im)));
		diff[index++] = std::accumulate(im.kern.begin(), im.kern.end(), 0.0f);
	}
	grad->descent(bias, diff);
	return next_loss;
}


convolution_layer::convolution_layer(size_t kerns, dim_2_sizes lin_size, 
	size_t in_depth) : lin_size(lin_size), out_depth(in_depth * kerns) {
	for (size_t i = 0; i < kerns; ++i) { kernels.push_back(*matrix::generate(lin_size)); }
	for (size_t i = 0; i < kerns; ++i) { flip_kernels.push_back(*matrix::flip(kernels[i])); }
	grads = std::vector<gradient>(kerns, gradient(lin_size.first * lin_size.second));
	next_tensor = std::make_shared<tensor>(out_depth);
	next_loss = std::make_shared<tensor>(in_depth);
	diff = std::make_shared<tensor>(kerns);
}

dim_2_sizes convolution_layer::prepare(size_t im_height, size_t im_width) {
	in_size = std::make_pair(im_height, im_width);
	out_size = std::make_pair(im_height - lin_size.first + 1, im_width - lin_size.second + 1);
	extend = std::make_pair(2 * lin_size.first - 2, 2 * lin_size.second - 2);
	return out_size;
}

tensor_ptr convolution_layer::forward(const tensor_ptr& prev_tensor) {
	last_input = tensor_ptr(prev_tensor);
	next_tensor->kern.clear();
	for (const matrix & im : prev_tensor->kern) {
		for (const matrix& kernel : kernels) { next_tensor->kern.push_back(std::move(*convolve(im, kernel, out_size))); }
	}
	return next_tensor;
}

tensor_ptr convolution_layer::fast_forward(const tensor& prev_tensor) {
	next_tensor->kern.clear();
	for (const matrix& im : prev_tensor.kern) {
		for (const matrix& kernel : kernels) { next_tensor->kern.push_back(std::move(*convolve(im, kernel, out_size))); }
	}
	return next_tensor;
}

matrix_ptr convolution_layer::convolve(const matrix& im, const matrix& kernel, dim_2_sizes new_size) {
	matrix_ptr convoluted = std::make_shared<matrix>(new_size);
	for (size_t y = 0; y < new_size.first; ++y) {
		for (size_t x = 0; x < new_size.second; ++x) {
			for (size_t i = 0; i < kernel.size.first; ++i) {
				for (size_t j = 0; j < kernel.size.second; ++j) { 
					convoluted->at(y, x) += im.at(y + i, x + j) * kernel.at(i, j); 
				}
			}
		}
	}
	return convoluted;
}

tensor_ptr convolution_layer::backprop(const tensor& prev_loss) {
	next_loss->kern.clear(), diff->kern.clear();
	for (size_t i = 0; i < kernels.size(); ++i) { diff->kern.push_back(matrix(lin_size)); }
	for (size_t i = 0; i < prev_loss.depth / kernels.size(); ++i) { next_loss->kern.push_back(matrix(in_size)); }
	for (size_t im = 0; im < prev_loss.depth; ++im) {
		size_t ker = im % kernels.size(), image = im / kernels.size();
		matrix_ptr ext_im = make_padding(prev_loss.kern[im], extend);
		diff->kern[ker] += *convolve(last_input->kern[image], *matrix::flip(prev_loss.kern[im]), lin_size);
		next_loss->kern[image] += *convolve(*ext_im, flip_kernels[ker], in_size);
	}
	for (size_t ker = 0; ker < kernels.size(); ++ker) { grads[ker].descent(kernels[ker].kern, diff->kern[ker].kern); }
	for (size_t ker = 0; ker < kernels.size(); ++ker) { flip_kernels[ker] = std::move(*matrix::flip(kernels[ker])); }
	last_input.reset();
	return next_loss;
}


pooling_layer::pooling_layer(size_t stride) : stride(stride) {}

dim_2_sizes pooling_layer::prepare(size_t im_height, size_t im_width, size_t depth) { 
	in_size = std::make_pair(im_height, im_width);
	miss.first = (stride - im_height % stride) % stride;
	miss.second = (stride - im_height % stride) % stride;
	out_size.first = (im_height + miss.first) / stride;
	out_size.second = (im_width + miss.second) / stride;
	maximals = std::vector<map_maximals>(depth, map_maximals(out_size.first * out_size.second));
	next_tensor = std::make_shared <tensor>(depth);
	next_loss = std::make_shared <tensor>(depth);
	return out_size;
}

matrix_ptr pooling_layer::pool(const matrix& im, size_t im_index) {
	matrix_ptr pooled = std::make_shared<matrix>(out_size);
	matrix_ptr extended = make_padding(im, miss);
	for (size_t y = 0; y < out_size.first; ++y) {
		for (size_t x = 0; x < out_size.second; ++x) {
			float local_max = std::numeric_limits<float>::min();
			for (size_t i = 0; i < stride; ++i) {
				for (size_t j = 0; j < stride; ++j) {
					size_t cur_y = y * stride + i, cur_x = x * stride + j;
					if (extended->at(cur_y, cur_x) > local_max) {
						maximals[im_index][y * out_size.second + x] = { cur_y, cur_x };
						local_max = extended->at(cur_y, cur_x);
					}
				}
			}
			pooled->at(y, x) = local_max;
		}
	}
	return pooled;
}

tensor_ptr pooling_layer::forward(const tensor_ptr& input_tensor) {
	next_tensor->kern.clear();
	for (size_t im = 0; im < input_tensor->depth; ++im) {
		next_tensor->kern.push_back(*pool(input_tensor->kern[im], im)); 
	}
	return next_tensor;
}

tensor_ptr pooling_layer::fast_forward(const tensor& input_tensor) {
	next_tensor->kern.clear();
	for (size_t im = 0; im < input_tensor.depth; ++im) {
		next_tensor->kern.push_back(std::move(*pool(input_tensor.kern[im], im)));
	}
	return next_tensor;
}

tensor_ptr pooling_layer::backprop(const tensor& prev_loss) {
	next_loss->kern.clear();
	for (size_t im = 0; im < prev_loss.depth; ++im) {
		matrix unpooled(in_size);
		for (size_t y = 0; y < out_size.first; ++y) {
			for (size_t x = 0; x < out_size.second; ++x) {
				dim_2_sizes coord_max = maximals[im][y * out_size.second + x];
				unpooled.at(coord_max.first, coord_max.second) = prev_loss.kern[im].at(y, x);
			}
		}
		next_loss->kern.push_back(std::move(unpooled));
	}
	return next_loss;
}
