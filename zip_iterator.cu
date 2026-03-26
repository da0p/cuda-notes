#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/universal_vector.h>
#include <thrust/zip_function.h>

int main()
{
    // allocate and initialize input vectors
    thrust::cuda::universal_vector<float> a{ 31, 22, 35 };
    thrust::cuda::universal_vector<float> b{ 25, 21, 27 };

    // zip two vectors into a single iterator
    auto zip = thrust::make_zip_iterator(a.begin(), b.begin());

    thrust::tuple<float, float> first = *zip;
    std::printf("first: (%g, %g)\n", thrust::get<0>(first), thrust::get<1>(first));

    zip++;

    thrust::tuple<float, float> second = *zip;
    std::printf("second: (%g, %g)\n", thrust::get<0>(second), thrust::get<1>(second));
}