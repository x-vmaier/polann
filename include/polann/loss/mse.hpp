#pragma once

#include <span>
#include "polann/config.h"
#ifdef POLANN_ENABLE_AVX2
#include <immintrin.h>
#endif

namespace polann::loss
{
    struct MSE
    {
        [[nodiscard]] inline static float compute(
            const std::span<const float> &yPred,
            const std::span<const float> &yTrue)
        {
            if (yPred.size() != yTrue.size())
                throw std::runtime_error("MSE requires spans of equal size");

            // Initialize params
            const std::size_t n = yPred.size();
            std::size_t i = 0;
            float sum = 0.0f;

#ifdef POLANN_ENABLE_AVX2
            __m256 vsum = _mm256_setzero_ps();
            for (; i + 8 <= n; i += 8)
            {
                __m256 va = _mm256_loadu_ps(yPred.data() + i);
                __m256 vb = _mm256_loadu_ps(yTrue.data() + i);
                __m256 vdiff = _mm256_sub_ps(va, vb);
                vsum = _mm256_fmadd_ps(vdiff, vdiff, vsum); // FMA: (a-b)^2 + acc
            }

            // Reduce 256 to 128 to scalar
            __m128 low = _mm256_castps256_ps128(vsum);
            __m128 high = _mm256_extractf128_ps(vsum, 1);
            __m128 sum128 = _mm_add_ps(low, high);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);

            sum = _mm_cvtss_f32(sum128);
#endif
            // Scalar remainder
            for (; i < n; ++i)
            {
                float diff = yPred[i] - yTrue[i];
                sum += diff * diff;
            }

            return sum / static_cast<float>(n);
        }

        inline static void grad(
            const std::span<const float> &yPred,
            const std::span<const float> &yTrue,
            std::span<float> gradOut)
        {
            if (gradOut.size() != yPred.size())
                throw std::runtime_error("Gradient output span must match prediction size");

            // Initialize params
            const std::size_t n = yPred.size();
            const float invN = 2.0f / n;
            size_t i = 0;

#ifdef POLANN_ENABLE_AVX2
            __m256 vInvN = _mm256_set1_ps(invN);
            for (; i + 8 <= n; i += 8)
            {
                __m256 vPred = _mm256_loadu_ps(yPred.data() + i);
                __m256 vTrue = _mm256_loadu_ps(yTrue.data() + i);
                __m256 vDiff = _mm256_sub_ps(vPred, vTrue); // yPred - yTrue
                __m256 vGrad = _mm256_mul_ps(vDiff, vInvN); // * (2/n)
                _mm256_storeu_ps(gradOut.data() + i, vGrad);
            }
#endif
            // Scalar remainder
            for (; i < n; ++i)
                gradOut[i] = invN * (yPred[i] - yTrue[i]);
        }
    };

} // namespace polann::loss
