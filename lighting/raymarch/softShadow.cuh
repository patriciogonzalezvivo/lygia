#include "map.cuh"

/*
contributors:  Inigo Quiles
description: Calculate soft shadows http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
use: <float> raymarchSoftshadow( in <float3> ro, in <float3> rd, in <float> tmin, in <float> tmax) 
*/

#ifndef RAYMARCHSOFTSHADOW_ITERATIONS
#define RAYMARCHSOFTSHADOW_ITERATIONS 16
#endif

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC(POS) raymarchMap(POS)
#endif

#ifndef RAYMARCH_MAP_DISTANCE
#define RAYMARCH_MAP_DISTANCE w
#endif

#ifndef FNC_RAYMARCH_SOFTSHADOW
#define FNC_RAYMARCH_SOFTSHADOW

inline __host__ __device__ float raymarchSoftShadow(const float3& ro, const float3& rd, float tmin, float tmax, float k ) {
    float res = 1.0f;
    float t = tmin;
    float ph = 1e20;
    for (int i = 0; i < RAYMARCHSOFTSHADOW_ITERATIONS; i++) {
        float h = RAYMARCH_MAP_FNC(ro + rd*t).RAYMARCH_MAP_DISTANCE;

        if (t > tmax)
            break;

        else if (h < 0.001) {
            res = 0.0;
            break;
        }

        float y = h * h / (2.0f * ph);
        float d = sqrt( h * h - y * y);
        res = min( res, k * d / max(0.0f, t - y) );
        ph = h;
        t += h;
    }
    return res;
}

inline __host__ __device__ float raymarchSoftShadow(const float3& ro, const float3& rd, float tmin, float tmax) { return raymarchSoftShadow(ro, rd, tmin, tmax, 12.0f); }
inline __host__ __device__ float raymarchSoftShadow(const float3& ro, const float3& rd) { return raymarchSoftShadow(ro, rd, 0.05f, 5.0f); }

#endif