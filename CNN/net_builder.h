#pragma once
#include"fcn.h"
#include"cnn.h"
#include<memory>
#include<string>

struct net_builder {
	using shared_cnn = std::shared_ptr<convolutional_network>;
	using shared_fcn = std::shared_ptr<fully_connected_network>;

private:
	using view = std::vector<std::string>;
	struct fcn {
		fcn(shared_cnn owner, view net_view);

		shared_fcn net;
		shared_cnn owner;

		fcn& hidden(size_t neurons);
		fcn& with_relu(float alpha = 0.05);
		fcn& with_tanh();
		fcn& with_sigmoid();
		fcn& with_softplus();
		fcn& last();

		convolutional_network&& build();

	private:
		view net_view;
	};

	struct cnn {
		cnn(size_t im_height, size_t im_width, size_t labels);
		shared_cnn net;

		cnn& conv(size_t kernels = 15, size_t y = 5, size_t x = 5);
		cnn& pool(size_t stride = 2);
		cnn& relu(float alpha = 0.05);
		cnn& tanh();
		cnn& sigmoid();
		cnn& softplus();

		fcn full_con();
	private:
		view net_view;
	};

public:
	cnn request(size_t im_height, size_t im_width, size_t labels);
};