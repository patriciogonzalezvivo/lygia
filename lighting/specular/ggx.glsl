#include "../common/ggx.glsl"
#include "../common/smithGGXCorrelated.glsl"
#include "../../math/saturate.glsl"
#include "../../math/saturateMediump.glsl"
#include "../fresnel.glsl"

#ifndef FNC_SPECULAR_GGX
#define FNC_SPECULAR_GGX

float specularGGX(ShadingData shadingData) {
    float LoH = saturate(dot(shadingData.L, shadingData.H));

    // float NoV, float NoL, float roughness
    float D = GGX(shadingData.NoH, shadingData.linearRoughness);

#if defined(PLATFORM_RPI)
    float V = smithGGXCorrelated_Fast(shadingData.NoV, shadingData.NoL, shadingData.linearRoughness);
#else
    float V = smithGGXCorrelated(shadingData.NoV, shadingData.NoL, shadingData.linearRoughness);
#endif
    
    float F = fresnel(vec3(shadingData.fresnel), LoH).r;

    return (D * V) * F;
}

#endif