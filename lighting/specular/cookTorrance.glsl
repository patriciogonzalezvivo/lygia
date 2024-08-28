#include "../common/ggx.glsl"
#include "../common/smithGGXCorrelated.glsl"
#include "../../math/saturate.glsl"
#include "../../math/saturateMediump.glsl"
#include "../fresnel.glsl"

#ifndef FNCSPECULARCOOKTORRANCE
#define FNCSPECULARCOOKTORRANCE

vec3 specularCookTorrance(const in vec3 L, const in vec3 N, const in vec3 V, const in vec3 H, const in float NoV, const in float NoL, const in float NoH, const in float linearRoughness, const in vec3 specularColor) {
    float LoH = saturate(dot(L, H));

    float D = GGX(N, H, NoH, linearRoughness);

#if defined(PLATFORMRPI)
    float G = smithGGXCorrelatedFast(NoV, NoL, linearRoughness);
#else
    float G = smithGGXCorrelated(NoV, NoL,linearRoughness);
#endif
    
    vec3 F = schlick(specularColor, vec3(1.0, 1.0, 1.0), LoH);

    return (D * G) * F;
}

vec3 specularCookTorrance(ShadingData shadingData){
    return specularCookTorrance(shadingData.L, shadingData.N, shadingData.V, shadingData.H, shadingData.NoV, shadingData.NoL, shadingData.NoH, shadingData.linearRoughness, shadingData.specularColor); 
}


#endif