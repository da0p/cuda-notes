#include <cstdlib>
#include <cuda_runtime.h>
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

__global__ void kernel_1(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

__global__ void kernel_2(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    std::cout << "Stream callback: Operation completed\n";
}

int main(void) {
    const int N = 1000000;
    size_t size = N * sizeof(float);
    float *h_data, *d_data;
    cudaStream_t stream_1, stream_2;
    cudaEvent_t event;

    // allocate host and device memory
    CHECK_CUDA_ERROR(cudaMallocHost(&h_data, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size));

    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // create streams with different priorities
    int leastPriority, greatestPriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream_1, cudaStreamNonBlocking, leastPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream_2, cudaStreamNonBlocking, greatestPriority));

    // create event
    CHECK_CUDA_ERROR(cudaEventCreate(&event));

    // asynchronous memory copy and kernel execution in stream_1
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream_1));
    kernel_1<<<(N + 255) / 256, 256, 0, stream_1>>>(d_data, N);

    // record event in stream_1
    CHECK_CUDA_ERROR(cudaEventRecord(event, stream_1));

    // make stream_2 wait for event
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_2, event, 0));
    // execute kernel in stream_2
    kernel_2<<<(N + 255) / 256, 256,0 , stream_2>>>(d_data, N);
    // add callback to stream_2
    CHECK_CUDA_ERROR(cudaStreamAddCallback(stream_2, myStreamCallback, NULL, 0));

    // asynchronous memory copy back to host
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream_2));

    // synchronize streams
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_2));

    // verify result
    for (int i = 0; i < N; i++) {
        float expected = (static_cast<float>(i) * 2.0f) + 1.0f;
        if (fabs(h_data[i] - expected) > 1e-5) {
            auto error = std::format("Result verification failed at element {}!", i);
            std::cout << error << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << std::format("Test PASSED") << std::endl;

    // clean up
    CHECK_CUDA_ERROR(cudaFreeHost(h_data));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_2));
    CHECK_CUDA_ERROR(cudaEventDestroy(event));
}