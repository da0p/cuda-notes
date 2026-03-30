#include <thrust/tabulate.h>
#include <thrust/universal_vector.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>
#include <iostream>

__host__ __device__
std::pair<int, int> row_col(int id, int width) {
    return std::make_pair(id / width, id % width);
}

void simulate(int height, int width,
    const thrust::universal_vector<float> &in, thrust::universal_vector<float> &out)
{
    const float *in_ptr = thrust::raw_pointer_cast(in.data());

    thrust::tabulate(
        thrust::device, out.begin(), out.end(),
        [in_ptr, height, width] __host__ __device__(int id) {
            auto [row, column] = row_col(id, width);

            if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
                float d2tdx2 =
                   in_ptr[(row) * width + column - 1] - 2 * in_ptr[row * width + column] + in_ptr[(row) * width + column + 1];
                float d2tdy2 =
                   in_ptr[(row - 1) * width + column] - 2 * in_ptr[row * width + column] + in_ptr[(row + 1) * width + column];

                return in_ptr[row * width + column] + 0.2f * (d2tdx2 + d2tdy2);
            } else {
                return in_ptr[row * width + column];
            }
        }
    );
}

int main() {
    const int height = 6;
    const int width = 6;
    const int size = height * width;

    thrust::universal_vector<float> in(size, 0.0f);
    thrust::universal_vector<float> out(size, 0.0f);

    // Hot spot in the center
    in[3 * width + 3] = 100.0f;

    const int steps = 100;
    for (int i = 0; i < steps; ++i) {
        simulate(height, width, in, out);
        thrust::swap(in, out);
    }

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c)
            std::cout << in[r * width + c] << "\t";
        std::cout << "\n";
    }
}