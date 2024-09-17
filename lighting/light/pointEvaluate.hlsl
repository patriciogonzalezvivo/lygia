/*
contributors: [Patricio Gonzalez Vivo, Shadi El Hajj]
description: Calculate point light
use: <void> lightPointEvaluate(<LightPoint> L, <Material> mat, inout <ShadingData> shadingData)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SURFACE_POSITION: in glslViewer is v_position
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_COLOR: in glslViewer is u_lightColor
    - LIGHT_INTENSITY: in glslViewer is  u_lightIntensity
    - LIGHT_FALLOFF: in glslViewer is u_lightFalloff
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#include "../specular.hlsl"
#include "../diffuse.hlsl"
#include "falloff.hlsl"

#ifndef FNC_LIGHT_POINT
#define FNC_LIGHT_POINT

void lightPointEvaluate(LightPoint L, Material mat, inout ShadingData shadingData) {

    float Ldist  = length(L.position);
    float3 Ldirection = L.position/Ldist;
    shadingData.L = Ldirection;
    shadingData.H = normalize(Ldirection + shadingData.V);
    shadingData.NoL = saturate(dot(shadingData.N, Ldirection));
    shadingData.NoH = saturate(dot(shadingData.N, shadingData.H));

    #ifdef FNC_RAYMARCH_SOFTSHADOW    
    float shadow = raymarchSoftShadow(mat.position, Ldirection);
    #else
    float shadow = 1.0;
    #endif

    float dif  = diffuse(shadingData);
    float3 spec = specular(shadingData);

    float3 lightContribution = L.color * L.intensity * shadow * shadingData.NoL;
    if (L.falloff > 0.0)
        lightContribution *= falloff(Ldist, L.falloff);

    shadingData.directDiffuse  += max(float3(0.0, 0.0, 0.0), shadingData.diffuseColor * lightContribution * dif);
    shadingData.directSpecular += max(float3(0.0, 0.0, 0.0), lightContribution * spec) * shadingData.energyCompensation;

    // TODO:
    // - make sure that the shadow use a perspective projection
    #ifdef SHADING_MODEL_SUBSURFACE
    float scatterVoH = saturate(dot(shadingData.V, -Ldirection));
    float forwardScatter = exp2(scatterVoH * mat.subsurfacePower - mat.subsurfacePower);
    float backScatter = saturate(shadingData.NoL * mat.subsurfaceThickness + (1.0 - mat.subsurfaceThickness)) * 0.5;
    float subsurface = lerp(backScatter, 1.0, forwardScatter) * (1.0 - mat.subsurfaceThickness);
    shadingData.directDiffuse += mat.subsurfaceColor * (subsurface * diffuseLambertConstant());
    #endif
}

#endif
