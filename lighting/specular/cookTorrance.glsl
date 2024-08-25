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
#if defined(PLATFORM_RPI)
#define SPECULAR_COOKTORRANCE_DIFFUSE_FNC beckmann
#else
#define SPECULAR_COOKTORRANCE_DIFFUSE_FNC GGX
#endif
#endif 

#ifndef FNC_SPECULAR_COOKTORRANCE
#define FNC_SPECULAR_COOKTORRANCE
// https://github.com/glslify/glsl-specular-cook-torrance
float specularCookTorrance(ShadingData shadingData) {

    float VoH = dot(shadingData.V, shadingData.H);

    // Geometric term

    float x = 2.0 * shadingData.NoH / VoH;
    float G = min(1.0, min(x * shadingData.NoV, x * shadingData.NoL));
    
    // Distribution term
    float D = SPECULAR_COOKTORRANCE_DIFFUSE_FNC(shadingData.N, shadingData.H, shadingData.NoH, shadingData.linearRoughness);

    // Fresnel term
    float F = SPECULAR_POW(1.0 - shadingData.NoV, shadingData.fresnel);

    // Multiply terms and done
    return max(G * F * D / max(PI * shadingData.NoV * shadingData.NoL, 0.00001), 0.0);
}

#endif