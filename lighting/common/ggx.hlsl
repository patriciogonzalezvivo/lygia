#include "../../math/const.hlsl"
#include "../../math/saturateMediump.hlsl"

#ifndef FNC_COMMON_GGX
#define FNC_COMMON_GGX

// Walter et al. 2007, "Microfacet Models for Refraction through Rough Surfaces"

// In mediump, there are two problems computing 1.0 - NoH^2
// 1) 1.0 - NoH^2 suffers floating point cancellation when NoH^2 is close to 1 (highlights)
// 2) NoH doesn't have enough precision around 1.0
// Both problem can be fixed by computing 1-NoH^2 in highp and providing NoH in highp as well

// However, we can do better using Lagrange's identity:
//      ||a x b||^2 = ||a||^2 ||b||^2 - (a . b)^2
// since N and H are unit vectors: ||N x H||^2 = 1.0 - NoH^2
// This computes 1.0 - NoH^2 directly (which is close to zero in the highlights and has
// enough precision).
// Overall this yields better performance, keeping all computations in mediump

float GGX(float NoH, float linearRoughness) {
    float oneMinusNoHSquared = 1.0 - NoH * NoH;
    float a = NoH * linearRoughness;
    float k = linearRoughness / (oneMinusNoHSquared + a * a);
    float d = k * k * INV_PI;
    return saturateMediump(d);
}

float GGX(float3 N, float3 H, float NoH, float linearRoughness) {
    float3 NxH = cross(N, H);
    float oneMinusNoHSquared = dot(NxH, NxH);

    float a = NoH * linearRoughness;
    float k = linearRoughness / (oneMinusNoHSquared + a * a);
    float d = k * k * INV_PI;
    return saturateMediump(d);
}

float3 importanceSamplingGGX(float2 u, float roughness) {
    float a2 = roughness * roughness;
    float phi = 2.0 * PI * u.x;
    float cosTheta2 = (1.0 - u.y) / (1.0 + (a2 - 1.0) * u.y);
    float cosTheta = sqrt(cosTheta2);
    float sinTheta = sqrt(1.0 - cosTheta2);
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

#endif