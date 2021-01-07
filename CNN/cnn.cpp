#include"cnn.h"
#include<iostream>
#include<iomanip>
#include<algorithm>

convolutional_network::convolutional_network(size_t im_height, size_t im_width, size_t labels) :
	depth(1), height(im_height), width(im_width), unique(labels) {}

void convolutional_network::fit(const images& train, const labels& targets) {
	for (size_t x = 0, tp = 0; x < train.size(); ++x) {
		tensor_ptr sample = std::make_shared<tensor>(train[x]);
		for (std::unique_ptr<cnn_layer>& layer : layers) { sample = layer->forward(sample); }
		flat_ptr net_view = fcn->forward(flatten(*sample));
		auto y = std::max_element(net_view->begin(), net_view->end());
		if (y == net_view->begin() + targets[x]) { std::cout << ++tp << " : " << x << std::endl; }
		flat_ptr diff = fcn->backprop(targets[x], *net_view);
		tensor_ptr tensor_diff = deflate(*diff);
		for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer) {
			tensor_diff = (*layer)->backprop(*tensor_diff);
		}
	}
}

labels convolutional_network::predict(const images& test) {
	labels predicted(test.size());
	for (size_t x = 0; x < test.size(); ++x) {
		tensor_ptr sample = std::make_shared<tensor>(test[x]);
		for (std::unique_ptr<cnn_layer>& layer : layers) { sample = layer->fast_forward(*sample); }
		flat_ptr net_view = fcn->fast_forward(flatten(*sample));
		auto y = std::max_element(net_view->begin(), net_view->end());
		predicted[x] = y - net_view->begin();
		std::cout << x << " : " << std::setprecision(3) << *y << std::endl;
	}
	return predicted;
}

flat_ptr convolutional_network::flatten(const tensor& input_tensor) {
	flat_ptr vectorised = flat_share(out_tensor_size);
	size_t position = 0;
	for (const matrix& sample : input_tensor.kern) {
		for (float pix : sample.kern) { vectorised->at(position++) = pix; }
	}
	return vectorised;
}

tensor_ptr convolutional_network::deflate(const one_dim& vectorised) {
	tensor_ptr deflated = std::make_shared<tensor>(depth);
	size_t image_size = height * width;
	for (size_t z = 0; z < depth; ++z) {
		size_t offset = z * image_size;
		two_dim im_view(image_size);
		for (size_t w = 0; w < image_size; ++w) { im_view[w] = vectorised[offset + w]; }
		deflated->kern.emplace_back(std::move(im_view), std::make_pair(height, width));
	}
	return deflated;
}