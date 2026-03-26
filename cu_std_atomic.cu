#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <iostream>
#include <format>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        auto error = std::format("CUDA error at {}:{} code={}({}) \"{}\"", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        std::cout << error << std::endl;
        exit(EXIT_FAILURE);
    }
}

constexpr int NUM_BINS = 256;

// Each thread atomically increments a bin counter.
// cuda::atomic with thread_scope_device ensures all threads across all blocks
// see each other's updates — equivalent to atomicAdd() but C++ style.
__global__ void histogram(const unsigned char* data, int n,
                          cuda::atomic<int, cuda::thread_scope_device>* bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bins[data[idx]].fetch_add(1, cuda::std::memory_order_relaxed);
    }
}

int main() {
    const int N = 1 << 24; // 16M elements

    // generate data on host
    unsigned char* h_data = new unsigned char[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<unsigned char>(i % NUM_BINS);
    }

    // copy data to device
    unsigned char* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, N));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice));

    // allocate bins on device as cuda::atomic array (zero-initialized)
    cuda::atomic<int, cuda::thread_scope_device>* d_bins;
    CHECK_CUDA_ERROR(cudaMalloc(&d_bins, NUM_BINS * sizeof(cuda::atomic<int, cuda::thread_scope_device>)));
    CHECK_CUDA_ERROR(cudaMemset(d_bins, 0, NUM_BINS * sizeof(cuda::atomic<int, cuda::thread_scope_device>)));

    histogram<<<(N + 255) / 256, 256>>>(d_data, N, d_bins);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // copy bins back and verify
    int h_bins[NUM_BINS];
    CHECK_CUDA_ERROR(cudaMemcpy(h_bins, d_bins, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    int expected = N / NUM_BINS;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (h_bins[i] != expected) {
            std::cout << std::format("Mismatch at bin {}: got {}, expected {}\n", i, h_bins[i], expected);
            exit(EXIT_FAILURE);
        }
    }

    std::cout << std::format("Histogram PASSED: {} bins, {} counts each\n", NUM_BINS, expected);

    delete[] h_data;
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_bins));
}
