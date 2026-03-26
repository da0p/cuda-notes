#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/universal_vector.h>

float median(thrust::universal_vector<float> vec) {
    thrust::sort(thrust::device, vec.begin(), vec.end());
    return vec[vec.size() / 2];
}

int main() {
    float k = 0.5;
    float ambient_temp = 20;
    thrust::universal_vector<float> temp{ 42, 24, 50 };
    auto transformation = [=] __host__ __device__ (float temp) {return temp + k * (ambient_temp - temp);};

    std::printf("temp median\n");
    for (int step = 0; step < 3; step++) {
        thrust::transform(thrust::device, temp.begin(), temp.end(), temp.begin(), transformation);
        float median_temp = median(temp);
        std::printf("%d        %.2f\n", step, median_temp);
    }

    return 0;
}