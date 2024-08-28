#include "../common/ggx.glsl"
#include "../common/smithGGXCorrelated.glsl"
#include "../../math/saturate.glsl"
#include "../../math/saturateMediump.glsl"
#include "../fresnel.glsl"

#ifndef FNC_SPECULAR_COOKTORRANCE
#define FNC_SPECULAR_COOKTORRANCE

float specularCookTorrance(const in vec3 _L, const in vec3 _N, const in vec3 _V, const in vec3 H, const in float _NoV, const in float _NoL, const in float _NoH, const in float _roughness, const in float _fresnel) {
    float NoL = max(_NoL, 0.0);
    float NoH = max(_NoH, 0.0);
    float LoH = saturate(dot(_L, H));

    float linearRoughness =  _roughness * _roughness;
    float D = GGX(NoH, linearRoughness);

#if defined(PLATFORM_RPI)
    float V = smithGGXCorrelated_Fast(_NoV, NoL,linearRoughness);
#else
    float V = smithGGXCorrelated(_NoV, NoL,linearRoughness);
#endif
    
    float F = fresnel(vec3(_fresnel, _fresnel, _fresnel), LoH).r;

    return (D * V) * F;
}

float specularCookTorrance(ShadingData shadingData){
    return specularCookTorrance(shadingData.L, shadingData.N, shadingData.V, shadingData.H, shadingData.NoV, shadingData.NoL, shadingData.NoH, shadingData.roughness, shadingData.fresnel); 
}


#endif