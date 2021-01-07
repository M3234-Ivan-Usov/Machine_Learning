#include"base.h"
#include<numeric>

flat_ptr flat_share(size_t share_size) { return std::make_shared<one_dim>(one_dim(share_size)); }

void reset_to_zero(one_dim& arr) { for (size_t pix = 0; pix < arr.size(); ++pix) { arr[pix] = 0.0f; } }

flat_ptr generate_flat(size_t size) { 
	std::mt19937 mt(rd());
	float xavier_init = sqrtf(6.0f) / (size * 2);
	std::uniform_real_distribution<> randomiser(-1.0 / size, 1.0 / size);
	flat_ptr generated = flat_share(size);
	for (float& el : *generated) { el = static_cast<float>(randomiser(mt)); }
	return generated;
}

matrix::matrix(two_dim&& kern, dim_2_sizes size) : kern(kern), size(size) {}

matrix::matrix(dim_2_sizes size) :size(size), kern(two_dim(size.first * size.second, 0.0f)) {}

float& matrix::at(size_t y, size_t x) { return kern[y * size.second + x]; }

float matrix::at(size_t y, size_t x) const { return kern[y * size.second + x]; }

matrix& matrix::operator+=(const matrix& other) {
	for (size_t pix = 0; pix < size.first * size.second; ++pix) { kern[pix] += other.kern[pix]; }
	return *this;
}

matrix_ptr matrix::generate(dim_2_sizes size) {
	std::mt19937 mt(rd());
	float xavier_init = sqrtf(6.0f) / (size.first + size.second);
	std::uniform_real_distribution<> randomiser(-xavier_init, xavier_init);
	matrix_ptr generated = std::make_shared<matrix>(size);
	for (auto& elem : generated->kern) { elem = static_cast<float>(randomiser(mt)); }
	return generated;
}

matrix_ptr matrix::flip(const matrix& im) {
	matrix_ptr transpose_first = matrix::transpose(im);
	matrix_ptr reverse = matrix::reverse_coloumns(*transpose_first);
	matrix_ptr transpose_second = matrix::transpose(*reverse);
	return matrix::reverse_coloumns(*transpose_second);
}

matrix_ptr matrix::transpose(const matrix& im) {
	matrix_ptr transposed = std::make_shared<matrix>(im.size);
	for (size_t y = 0; y < im.size.first; ++y) {
		for (size_t x = 0; x < im.size.second; ++x) {
			transposed->at(y, x) = im.at(x, y);
		}
	}
	std::swap(transposed->size.first, transposed->size.second);
	return transposed;
}

matrix_ptr matrix::reverse_coloumns(const matrix& im) {
	matrix_ptr reversed = std::make_shared<matrix>(im.size);
	for (size_t y = 0; y < im.size.first; ++y) {
		for (size_t x = 0; x < im.size.second; ++x) {
			reversed->at(y, x) = im.at(im.size.first - 1 - y, x);
		}
	}
	return reversed;
}

matrix_ptr matrix::scalar_dot(const matrix& other) {
	matrix_ptr scalared = std::make_shared<matrix>(size);
	for (size_t pix = 0; pix < size.first * size.second; ++pix) {
		scalared->kern[pix] = kern[pix] * other.kern[pix];
	}
	return scalared;
}

tensor::tensor(size_t depth) : depth(depth) { kern.reserve(depth); }

tensor::tensor(const matrix& single) : depth(1) { kern.push_back(single); }
