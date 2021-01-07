#include "grad.h"

gradient::gradient(size_t weights_amount) :
	m(std::vector<float>(weights_amount)), v(std::vector<float>(weights_amount)) {}

void gradient::descent(two_dim& weights, const two_dim& diffs) {
	b1_pow *= beta_1, b2_pow *= beta_2;
	for (size_t link = 0; link < weights.size(); ++link) {
		m[link] = beta_1 * m[link] + (1 - beta_1) * diffs[link];
		v[link] = beta_2 * v[link] + (1 - beta_2) * diffs[link] * diffs[link];
		float m_norm = m[link] / (1 - b1_pow), v_norm = v[link] / (1 - b2_pow);
		weights[link] -= speed / sqrtf(v_norm + eps) * m_norm;
	}
}
