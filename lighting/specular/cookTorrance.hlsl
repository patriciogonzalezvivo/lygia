#include "../common/ggx.hlsl"
#include "../common/smithGGXCorrelated.hlsl"
#include "../../math/saturateMediump.hlsl"
#include "../fresnel.hlsl"

#ifndef FNC_SPECULAR_COOKTORRANCE
#define FNC_SPECULAR_COOKTORRANCE

float specularCookTorrance(float3 _L, float3 _N, float3 _V, const in float3 H, float _NoV, float _NoL, const in float _NoH, float _roughness, float _fresnel) {
    float NoL = max(_NoL, 0.0);
    float LoH = saturate(dot(_L, H));
    float NoH = saturate(dot(_N, H));

    float linearRoughness =  _roughness * _roughness;
    float D = GGX(NoH, linearRoughness);

#if defined(PLATFORM_RPI)
    float V = smithGGXCorrelated_Fast(_NoV, NoL,linearRoughness);
#else
    float V = smithGGXCorrelated(_NoV, NoL,linearRoughness);
#endif
    
    float F = fresnel(float3(_fresnel, _fresnel, _fresnel), LoH).r;

    return (D * V) * F;
}

float specularCookTorrance(ShadingData shadingData){
    return specularCookTorrance(shadingData.L, shadingData.N, shadingData.V, shadingData.H, shadingData.NoV, shadingData.NoL, shadingData.NoH, shadingData.roughness, shadingData.fresnel); 
}

#endif