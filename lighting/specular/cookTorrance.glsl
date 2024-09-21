#include "../common/ggx.glsl"
#include "../common/smithGGXCorrelated.glsl"
#include "../../math/saturate.glsl"
#include "../../math/saturateMediump.glsl"
#include "../fresnel.glsl"

#ifndef FNC_SPECULAR_COOKTORRANCE
#define FNC_SPECULAR_COOKTORRANCE

vec3 specularCookTorrance(const in vec3 L, const in vec3 N, const in vec3 H, const in float NoV, const in float NoL, const in float NoH, const in float linearRoughness, const in vec3 specularColor) {
    float LoH = saturate(dot(L, H));

    float D = GGX(N, H, NoH, linearRoughness);

#if defined(PLATFORM_RPI)
    float V = smithGGXCorrelated_Fast(NoV, NoL, linearRoughness);
#else
    float V = smithGGXCorrelated(NoV, NoL,linearRoughness);
#endif
    
    vec3 F = fresnel(specularColor, LoH);

    return (D * V) * F;
}

vec3 specularCookTorrance(ShadingData shadingData){
    return specularCookTorrance(shadingData.L, shadingData.N, shadingData.H, shadingData.NoV, shadingData.NoL, shadingData.NoH, shadingData.linearRoughness, shadingData.specularColor); 
}


#endif