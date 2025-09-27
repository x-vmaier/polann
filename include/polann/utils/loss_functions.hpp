#pragma once

#include <span>
#include <immintrin.h>

namespace polann::utils::loss_functions
{
    [[nodiscard]] inline float mse(std::span<const float> a, std::span<const float> b)
    {
        if (a.size() != b.size())
            throw std::runtime_error("MSE requires spans of equal size");

        const std::size_t n = a.size();
        __m256 vsum = _mm256_setzero_ps();

        std::size_t i = 0;
        for (; i + 8 <= n; i += 8)
        {
            __m256 va = _mm256_loadu_ps(a.data() + i);
            __m256 vb = _mm256_loadu_ps(b.data() + i);
            __m256 vdiff = _mm256_sub_ps(va, vb);
            vsum = _mm256_fmadd_ps(vdiff, vdiff, vsum); // FMA: (a-b)^2 + acc
        }

        // Reduce 256 to 128 to scalar
        __m128 low = _mm256_castps256_ps128(vsum);
        __m128 high = _mm256_extractf128_ps(vsum, 1);
        __m128 sum128 = _mm_add_ps(low, high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);

        float sum = _mm_cvtss_f32(sum128);

        // Scalar remainder
        for (; i < n; ++i)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }

        return sum / static_cast<float>(n);
    }

} // namespace polann::utils::loss_functions
