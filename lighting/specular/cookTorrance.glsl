#include "../common/beckmann.glsl"
#include "../common/ggx.glsl"
#include "../../math/powFast.glsl"
#include "../../math/saturate.glsl"
#include "../../math/const.glsl"

#ifndef SPECULAR_POW
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) || defined(PLATFORM_WEBGL)
#define SPECULAR_POW(A,B) powFast(A,B)
#else
#define SPECULAR_POW(A,B) pow(A,B)
#endif
#endif

#ifndef SPECULAR_COOKTORRANCE_DIFFUSE_FNC
#define SPECULAR_COOKTORRANCE_DIFFUSE_FNC GGX
// #define SPECULAR_COOKTORRANCE_DIFFUSE_FNC beckmann
#endif 

#ifndef FNC_SPECULAR_COOKTORRANCE
#define FNC_SPECULAR_COOKTORRANCE

// https://github.com/stackgl/glsl-specular-cook-torrance
float specularCookTorrance(vec3 _L, vec3 _N, vec3 _V, float _NoV, float _NoL, float _roughness, float _fresnel) {
    float NoV = max(_NoV, 0.0);
    float NoL = max(_NoL, 0.0);

    // Half angle vector
    vec3 H = normalize(_L + _V);

    // Geometric term
    float NoH = max(dot(_N, H), 0.0);
    float VoH = max(dot(_V, H), 0.000001);

    float x = 2.0 * NoH / VoH;
    float G = min(1.0, min(x * NoV, x * NoL));
    
    // Distribution term
    float D = SPECULAR_COOKTORRANCE_DIFFUSE_FNC(_N, H, NoH, _roughness);

    // Fresnel term
    float F = SPECULAR_POW(1.0 - NoV, _fresnel);

    // Multiply terms and done
    return max(G * F * D / max(PI * NoV * NoL, 0.00001), 0.0);
}

// https://github.com/glslify/glsl-specular-cook-torrance
float specularCookTorrance(vec3 L, vec3 N, vec3 V, float roughness, float fresnel) {
    return specularCookTorrance(L, N, V, dot(N, V), dot(N, L), roughness, fresnel);
}

float specularCookTorrance(vec3 L, vec3 N, vec3 V, float roughness) {
    return specularCookTorrance(L, N, V, roughness, 0.04);
}

#endif