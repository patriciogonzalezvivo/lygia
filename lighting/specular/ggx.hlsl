// #include "../common/ggx.hlsl"
#include "../common/smithGGXCorrelated.hlsl"
#include "../fresnel.hlsl"

#ifndef FNC_SPECULAR_GGX
#define FNC_SPECULAR_GGX

float commonGGX(float NoH, float linearRoughness) {
    float oneMinusNoHSquared = 1.0 - NoH * NoH;
    float a = NoH * linearRoughness;
    float k = linearRoughness / (oneMinusNoHSquared + a * a);
    float d = k * k * INV_PI;
    return saturateMediump(d);
}

float specularGGX(float3 _L, float3 _N, float3 _V, float _NoV, float _NoL, float _roughness, float _fresnel) {
    float NoV = max(_NoV, 0.0);
    float NoL = max(_NoL, 0.0);

    float3 H = normalize(_L + _V);
    float LoH = saturate(dot(_L, H));
    float NoH = saturate(dot(_N, H));

    // float NoV, float NoL, float roughness
    float linearRoughness =  _roughness * _roughness;
    float D = commonGGX(NoH, linearRoughness);
    float V = smithGGXCorrelated(_NoV, NoL,linearRoughness);
    float F = fresnel(float3(_fresnel, _fresnel, _fresnel), LoH).r;

    return (D * V) * F;
}

float specularGGX(float3 L, float3 N, float3 V, float roughness, float fresnel) {
    float NoV = max(dot(N, V), 0.0);
    float NoL = max(dot(N, L), 0.0);
    return specularGGX(L, N, V, NoV, NoL, roughness, fresnel);
}

float specularGGX(float3 L, float3 N, float3 V, float roughness) {
    return specularGGX(L, N, V, roughness, 0.04);
}

#endif