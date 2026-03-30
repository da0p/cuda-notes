#include <thrust/tabulate.h>
#include <thrust/universal_vector.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>
#include <cuda/std/mdspan>

__host__ __device__
cuda::std::pair<int, int> row_col(int id, int width) {
    return cuda::std::make_pair(id / width, id % width);
}

void simulate(int height, int width,
              const thrust::universal_vector<float> &in,
              thrust::universal_vector<float> &out)
{
    cuda::std::mdspan md(thrust::raw_pointer_cast(in.data()), height, width);
    thrust::tabulate(
        thrust::device, out.begin(), out.end(),
        [md, height, width] __host__ __device__ (int id) {
            auto [row, column] = row_col(id, width);

            if (row > 0 && column > 0 && row < height - 1 && column < width - 1) {
                float d2tdx2 = md(row, column - 1) - 2 * md(row, column) + md(row, column + 1);
                float d2tdy2 = md(row - 1, column) - 2 * md(row, column) + md(row + 1, column);
                return md(row, column) + 0.2f * (d2tdx2 + d2tdy2);
            } else {
                return md(row, column);
            }
        }
    );
}

int main() {
    const int height = 5;
    const int width = 5;
    const int size = height * width;

    thrust::universal_vector<float> in(size, 0.0f);
    thrust::universal_vector<float> out(size, 0.0f);

    // Set a hot spot in the center
    in[2 * width + 2] = 100.0f;

    simulate(height, width, in, out);

    // Print result
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            printf("%.2f ", out[r * width + c]);
        }
        printf("\n");
    }

    return 0;
}