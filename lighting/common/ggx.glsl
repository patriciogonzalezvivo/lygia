#ifndef ONE_OVER_PI
#define ONE_OVER_PI 0.31830988618
#endif

#include "../../math/saturateMediump.glsl"

#ifndef FNC_GGX
#define FNC_GGX

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
    float d = k * k * ONE_OVER_PI;
    return saturateMediump(d);
}

float GGX(vec3 N, vec3 H, float NoH, float linearRoughness) {
    vec3 NxH = cross(N, H);
    float oneMinusNoHSquared = dot(NxH, NxH);

    float a = NoH * linearRoughness;
    float k = linearRoughness / (oneMinusNoHSquared + a * a);
    float d = k * k * ONE_OVER_PI;
    return saturateMediump(d);
}

#endif