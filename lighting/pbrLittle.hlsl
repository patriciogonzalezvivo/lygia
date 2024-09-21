#include "../math/powFast.hlsl"
#include "../color/tonemap/reinhard.hlsl"

#include "shadow.hlsl"
#include "material.hlsl"
#include "fresnelReflection.hlsl"
#include "sphericalHarmonics.hlsl"

#include "ior.hlsl"
#include "envMap.hlsl"
#include "diffuse.hlsl"
#include "specular.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple PBR shading model
use:
    - <float4> pbrLittle(<Material> material)
    - <float4> pbrLittle(<float4> albedo, <float3> normal, <float> roughness, <float> metallic [, <float3> f0] )
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughness (default on mobile)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef CAMERA_POSITION
#define CAMERA_POSITION float3(0.0, 0.0, -10.0)
#endif

#ifndef LIGHT_POSITION
#define LIGHT_POSITION  float3(0.0, 10.0, -50.0)
#endif

#ifndef LIGHT_COLOR
#define LIGHT_COLOR     float3(0.5, 0.5, 0.5)
#endif

#ifndef FNC_PBR_LITTLE
#define FNC_PBR_LITTLE

float4 pbrLittle(Material mat, ShadingData shadingData) {
    shadingDataNew(mat, shadingData);
    #ifdef LIGHT_DIRECTION
    shadingData.L = normalize(LIGHT_DIRECTION);
    #else
    shadingData.L = normalize(LIGHT_POSITION - mat.position);
    #endif
    shadingData.H = normalize(shadingData.L + shadingData.V);
    shadingData.NoL = saturate(dot(shadingData.N, shadingData.L));
    shadingData.NoH = saturate(dot(shadingData.N, shadingData.H));

    float notMetal = 1.0 - mat.metallic;
    float smoothness = 0.95 - saturate(mat.roughness);

    #if defined(LIGHT_SHADOWMAP) && defined(LIGHT_SHADOWMAP_SIZE) && defined(LIGHT_COORD)
    float shadow = shadow(LIGHT_SHADOWMAP, float2(LIGHT_SHADOWMAP_SIZE), (LIGHT_COORD).xy, (LIGHT_COORD).z);
    #elif defined(FNC_RAYMARCH_SOFTSHADOW)
    float shadow = raymarchSoftShadow(mat.position, shadingData.L);
    #else
    float shadow = 1.0;
    #endif

    // DIFFUSE
    float diff = diffuse(shadingData) * shadow;
    float3 spec = specular(shadingData) * shadow;

    float3 albedo = mat.albedo.rgb * diff;
// #ifdef SCENE_SH_ARRAY
    // _albedo.rgb = _albedo.rgb + tonemapReinhard( sphericalHarmonics(N) ) * 0.25;
// #endif

    // SPECULAR
    // This is a bit of a stylistic approach
    float specIntensity =   (0.04 * notMetal + 2.0 * mat.metallic) * 
                            saturate(-1.1 + shadingData.NoV + mat.metallic) * // Fresnel
                            (mat.metallic + smoothness * 4.0); // make smaller highlights brighter

    float3 ambientSpecular = tonemapReinhard( envMap(mat, shadingData) ) * specIntensity;
    ambientSpecular += fresnelReflection(mat, shadingData) * (1.0-mat.roughness);

    albedo = albedo.rgb * notMetal + ( ambientSpecular 
                    + LIGHT_COLOR * 2.0 * spec
                    ) * (notMetal * smoothness + albedo * mat.metallic);

    return float4(albedo, mat.albedo.a);
}

float4 pbrLittle(const in Material mat) {
    ShadingData shadingData = shadingDataNew();
    shadingData.V = normalize(CAMERA_POSITION - mat.position);
    return pbrLittle(mat, shadingData);
}

#endif