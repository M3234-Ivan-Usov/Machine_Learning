#pragma once
#include"base.h"

struct gradient {
	const float beta_1 = 0.9f, beta_2 = 0.999f;
	const float speed = 1e-3f, eps = 1e-8f;

	float b1_pow = 1.0f, b2_pow = 1.0f;
	std::vector<float> m, v;

	gradient(size_t weights_amount);
	void descent(two_dim& weights, const two_dim& diffs);
};