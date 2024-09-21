#include "../common/ggx.hlsl"
#include "../common/smithGGXCorrelated.hlsl"
#include "../../math/saturateMediump.hlsl"
#include "../fresnel.hlsl"

#ifndef FNC_SPECULAR_COOKTORRANCE
#define FNC_SPECULAR_COOKTORRANCE

float3 specularCookTorrance(float3 L, float3 N, const in float3 H, float NoV, float NoL, const in float NoH, float linearRoughness, float3 specularColor) {
    float LoH = saturate(dot(L, H));

    float D = GGX(N, H, NoH, linearRoughness);

#if defined(PLATFORM_RPI)
    float V = smithGGXCorrelated_Fast(NoV, NoL, linearRoughness);
#else
    float V = smithGGXCorrelated(NoV, NoL, linearRoughness);
#endif
    
    float3 F = fresnel(specularColor, LoH);

    return (D * V) * F;
}

float3 specularCookTorrance(ShadingData shadingData){
    return specularCookTorrance(shadingData.L, shadingData.N, shadingData.H, shadingData.NoV, shadingData.NoL, shadingData.NoH, shadingData.linearRoughness, shadingData.specularColor);
}

#endif