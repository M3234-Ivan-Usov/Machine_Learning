#pragma once
#include<random>
#include<vector>
#include<memory>

using one_dim = std::vector<float>;
using two_dim = std::vector<float>;
using dim_2_sizes = std::pair<size_t, size_t>;

struct matrix;
struct tensor;

static std::random_device rd;

using flat_ptr = std::shared_ptr<one_dim>;
using matrix_ptr = std::shared_ptr<matrix>;
using tensor_ptr = std::shared_ptr<tensor>;

flat_ptr flat_share(size_t share_size);
flat_ptr generate_flat(size_t size);
void reset_to_zero(one_dim& arr);

struct matrix {
	two_dim kern;
	dim_2_sizes size;

	matrix(two_dim&& kern, dim_2_sizes size);
	matrix(dim_2_sizes size);

	static matrix_ptr generate(dim_2_sizes size);
	static matrix_ptr flip(const matrix& im);
	static matrix_ptr transpose(const matrix& im);
	static matrix_ptr reverse_coloumns(const matrix& im);

	float& at(size_t y, size_t x);
	float at(size_t y, size_t x) const;

	matrix& operator+=(const matrix& other);
	matrix_ptr scalar_dot(const matrix& other);
};


struct tensor {
	std::vector<matrix> kern;
	size_t depth;

	tensor(size_t depth);
	tensor(const matrix& single);
};

