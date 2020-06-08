#include <iostream>
#include <thread>
#include <cmath>
#include <immintrin.h>

static const uint32_t xorshift_seed = 2463534242;

static const double eps = 1e-10;

inline double xorshift_rand(uint32_t &x) {
	x ^= x >> 13;
	x ^= x << 17;
	x ^= x >> 5;
	return 1. * x / ((uint32_t)(-1));
}

template <class T>
void do_not_opt_out(T &&x) {
	static auto ttid = std::this_thread::get_id();
	if (ttid == std::thread::id()) {
		const auto* p = &x;
		putchar(*reinterpret_cast<const char*>(p));
		std::abort();
	}
}

template <typename T>
double benchmark(T fn) {
	double voxel_size = 3;

	double corner[3];
	corner[0] = 10;
	corner[1] = 11;
	corner[2] = 12;

	double values[2][2][2][3];
	double dd = 100;
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 2; ++k) {
				values[i][j][k][0] = dd++;
				values[i][j][k][1] = dd++;
				values[i][j][k][2] = dd++;
			}
		}
	}

	size_t nr_runs = 1e8;
	double result = 0;

	auto t1 = std::chrono::high_resolution_clock::now();

	uint32_t rand_state = xorshift_seed;

	for (size_t i = 0; i < nr_runs; ++i) {
		double point[3];
		point[0] = corner[0] + voxel_size * xorshift_rand(rand_state);
		point[1] = corner[1] + voxel_size * xorshift_rand(rand_state);
		point[2] = corner[2] + voxel_size * xorshift_rand(rand_state);

		double r[3];
		fn(r, point, corner, voxel_size, values);

		result += r[0];
		result += r[1];
		result += r[2];
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

	do_not_opt_out(result);

	return dt * 1e-3;
}

template <typename T>
bool compare(T fn1, T fn2) {
	double voxel_size = 3;

	double corner[3];
	corner[0] = 10;
	corner[1] = 11;
	corner[2] = 12;

	double values[2][2][2][3];
	double dd = 100;
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 2; ++k) {
				values[i][j][k][0] = dd++;
				values[i][j][k][1] = dd++;
				values[i][j][k][2] = dd++;
			}
		}
	}

	double input[][3] = {
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{0, 1, 1},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0},
		{1, 1, 1},
		{0.1, 0.1, 0.1},
		{0.1, 0.1, 0.8},
		{0.1, 0.8, 0.1},
		{0.1, 0.8, 0.8},
		{0.8, 0.1, 0.1},
		{0.8, 0.1, 0.8},
		{0.8, 0.8, 0.1},
		{0.8, 0.8, 0.8},
	};

	for (size_t i = 0; i < sizeof(input) / sizeof(input[0]); ++i) {
		double point[3];
		for (int j = 0; j < 3; ++j)
			point[j] = corner[j] + voxel_size * input[i][j];

		double r1[3];
		fn1(r1, point, corner, voxel_size, values);

		double r2[3];
		fn2(r2, point, corner, voxel_size, values);

		if (fabs(r1[0] - r2[0]) < eps && fabs(r1[1] - r2[1]) < eps && fabs(r1[2] - r2[2]) < eps)
			continue;

		std::cerr << "\nDifferent values\n";
		std::cerr << "at:     " << input[i][0] << " " << input[i][1] << "  " << input[i][2] << "\n";
		std::cerr << "point:  " << point[0] << " " << point[1] << "  " << point[2] << "\n";
		std::cerr << "r1:     " << r1[0] << " " << r1[1] << "  " << r1[2] << "\n";
		std::cerr << "r2:     " << r2[0] << " " << r2[1] << "  " << r2[2] << "\n";

		return false;
	}

	return true;
}

void tri_interp_original(
	double result[3],
	const double point[3],
	const double corner[3],
	double voxel_size,
	const double values[2][2][2][3]
) {
	double p[3];
	for (int i = 0; i < 3; ++i)
		p[i] = (point[i] - corner[i]) / voxel_size;

	for (int i = 0; i < 3; ++i) {
		result[i] = 0;
		result[i] += values[0][0][0][i] * (1 - p[0]) * (1 - p[1]) * (1 - p[2]);
		result[i] += values[0][0][1][i] * (1 - p[0]) * (1 - p[1]) * (    p[2]);
		result[i] += values[0][1][0][i] * (1 - p[0]) * (    p[1]) * (1 - p[2]);
		result[i] += values[0][1][1][i] * (1 - p[0]) * (    p[1]) * (    p[2]);
		result[i] += values[1][0][0][i] * (    p[0]) * (1 - p[1]) * (1 - p[2]);
		result[i] += values[1][0][1][i] * (    p[0]) * (1 - p[1]) * (    p[2]);
		result[i] += values[1][1][0][i] * (    p[0]) * (    p[1]) * (1 - p[2]);
		result[i] += values[1][1][1][i] * (    p[0]) * (    p[1]) * (    p[2]);
	}
}

// https://en.wikipedia.org/wiki/Trilinear_interpolation
void my_tri_interp(
	double result[3],
	const double point[3],
	const double corner[3],
	double voxel_size,
	const double values[2][2][2][3]
) {
	double p[3];
	for (int i = 0; i < 3; ++i)
		p[i] = (point[i] - corner[i]) / voxel_size;

	for (int i = 0; i < 3; ++i) {
		// interpolation on x
		double c00 = values[0][0][0][i] * (1 - p[0]) + values[1][0][0][i] * p[0];
		double c01 = values[0][0][1][i] * (1 - p[0]) + values[1][0][1][i] * p[0];
		double c10 = values[0][1][0][i] * (1 - p[0]) + values[1][1][0][i] * p[0];
		double c11 = values[0][1][1][i] * (1 - p[0]) + values[1][1][1][i] * p[0];

		// interpolation on y
		double c0 = c00 * (1 - p[1]) + c10 * p[1];
		double c1 = c01 * (1 - p[1]) + c11 * p[1];

		// interpolation on z
		result[i] = c0 * (1 - p[2]) + c1 * p[2];
	}
}

// vectorize the whole `for` loop
void tri_interp_vector_for_loop(
	double result[3],
	const double point[3],
	const double corner[3],
	double voxel_size,
	const double values[2][2][2][3]
) {
	__m256d point_vector = {
		point[0],
		point[1],
		point[2],
		0
	};
	__m256d corner_vector = {
		corner[0],
		corner[1],
		corner[2],
		0
	};
	__m256d p = (point_vector - corner_vector) / voxel_size;

	__m256d c000 = { values[0][0][0][0], values[0][0][0][1], values[0][0][0][2], 0};
	__m256d c001 = { values[0][0][1][0], values[0][0][1][1], values[0][0][1][2], 0};
	__m256d c010 = { values[0][1][0][0], values[0][1][0][1], values[0][1][0][2], 0};
	__m256d c011 = { values[0][1][1][0], values[0][1][1][1], values[0][1][1][2], 0};
	__m256d c100 = { values[1][0][0][0], values[1][0][0][1], values[1][0][0][2], 0};
	__m256d c101 = { values[1][0][1][0], values[1][0][1][1], values[1][0][1][2], 0};
	__m256d c110 = { values[1][1][0][0], values[1][1][0][1], values[1][1][0][2], 0};
	__m256d c111 = { values[1][1][1][0], values[1][1][1][1], values[1][1][1][2], 0};

	// x
	__m256d c00 = c000 * (1 - p[0]) + c100 * p[0];
	__m256d c01 = c001 * (1 - p[0]) + c101 * p[0];
	__m256d c10 = c010 * (1 - p[0]) + c110 * p[0];
	__m256d c11 = c011 * (1 - p[0]) + c111 * p[0];

	// y
	__m256d c0 = c00 * (1 - p[1]) + c10 * p[1];
	__m256d c1 = c01 * (1 - p[1]) + c11 * p[1];

	// z
	__m256d c = c0 * (1 - p[2]) + c1 * p[2];

	result[0] = c[0];
	result[1] = c[1];
	result[2] = c[2];
}

// simplify the expression
void tri_interp_vector_for_loop_simplified(
	double result[3],
	const double point[3],
	const double corner[3],
	double voxel_size,
	const double values[2][2][2][3]
) {
	// Your code goes there

	__m256d point_vector = {
		point[0],
		point[1],
		point[2],
		0
	};
	__m256d corner_vector = {
		corner[0],
		corner[1],
		corner[2],
		0
	};
	__m256d p = (point_vector - corner_vector) / voxel_size;

	__m256d c000 = { values[0][0][0][0], values[0][0][0][1], values[0][0][0][2], 0};
	__m256d c001 = { values[0][0][1][0], values[0][0][1][1], values[0][0][1][2], 0};
	__m256d c010 = { values[0][1][0][0], values[0][1][0][1], values[0][1][0][2], 0};
	__m256d c011 = { values[0][1][1][0], values[0][1][1][1], values[0][1][1][2], 0};
	__m256d c100 = { values[1][0][0][0], values[1][0][0][1], values[1][0][0][2], 0};
	__m256d c101 = { values[1][0][1][0], values[1][0][1][1], values[1][0][1][2], 0};
	__m256d c110 = { values[1][1][0][0], values[1][1][0][1], values[1][1][0][2], 0};
	__m256d c111 = { values[1][1][1][0], values[1][1][1][1], values[1][1][1][2], 0};

	// x
	__m256d c00 = c000 + (c100 - c000) * p[0];
	__m256d c01 = c001 + (c101 - c001) * p[0];
	__m256d c10 = c010 + (c110 - c010) * p[0];
	__m256d c11 = c011 + (c111 - c011) * p[0];

	// y
	__m256d c0 = c00 + (c10 - c00) * p[1];
	__m256d c1 = c01 + (c11 - c01) * p[1];

	// z
	__m256d c = c0 + (c1 - c0) * p[2];

	result[0] = c[0];
	result[1] = c[1];
	result[2] = c[2];
}

int main()
{
	std::cerr << "tri_interp_original: ";
	std::cerr << benchmark(tri_interp_original); 
	std::cerr << " sec\n";

	std::cerr << "my_tri_interp: ";
	std::cerr << benchmark(my_tri_interp); 
	std::cerr << " sec\n";

	std::cerr << "tri_interp_vector_for_loop: ";
	std::cerr << benchmark(tri_interp_vector_for_loop); 
	std::cerr << " sec\n";

	std::cerr << "tri_interp_vector_for_loop_simplified: ";
	std::cerr << benchmark(tri_interp_vector_for_loop_simplified); 
	std::cerr << " sec\n";

	if (!compare(tri_interp_original, my_tri_interp))
		std::cerr << "my_tri_interp not valid\n";
	else
		std::cerr << "my_tri_interp valid\n";
	
	if (!compare(tri_interp_original, tri_interp_vector_for_loop))
		std::cerr << "tri_interp_vector_for_loop not valid\n";
	else
		std::cerr << "tri_interp_vector_for_loop valid\n";


	if (!compare(tri_interp_original, tri_interp_vector_for_loop_simplified))
		std::cerr << "tri_interp_vector_for_loop_simplified not valid\n";
	else
		std::cerr << "tri_interp_vector_for_loop_simplified valid\n";
	
	return 0;
}

