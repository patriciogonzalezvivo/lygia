#include "../common/ggx.hlsl"
#include "../common/smithGGXCorrelated.hlsl"
#include "../common/schlick.hlsl"
#include "../../math/saturateMediump.hlsl"
#include "../fresnel.hlsl"

#ifndef FNCSPECULARCOOKTORRANCE
#define FNCSPECULARCOOKTORRANCE

float3 specularCookTorrance(float3 L, float3 N, float3 V, const in float3 H, float NoV, float NoL, const in float NoH, float roughness, float3 specularColor) {
    float LoH = saturate(dot(L, H));

    float linearRoughness =  roughness * roughness;
    float D = GGX(N, H, NoH, linearRoughness);

#if defined(PLATFORMRPI)
    float G = smithGGXCorrelatedFast(NoV, NoL, linearRoughness);
#else
    float G = smithGGXCorrelated(NoV, NoL,linearRoughness);
#endif
    
    float3 F = schlick(specularColor, float3(1.0, 1.0, 1.0), LoH);

    return (D * G) * F;
}

float3 specularCookTorrance(ShadingData shadingData){
    return specularCookTorrance(shadingData.L, shadingData.N, shadingData.V, shadingData.H, shadingData.NoV, shadingData.NoL, shadingData.NoH, shadingData.roughness, shadingData.specularColor);
}

#endif