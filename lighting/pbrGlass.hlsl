#include "../color/tonemap.hlsl"

#include "material.hlsl"
#include "light/new.hlsl"
#include "envMap.hlsl"
#include "specular.hlsl"
#include "fresnelReflection.hlsl"
#include "transparent.hlsl"

#include "ior/2eta.hlsl"
#include "ior/2f0.hlsl"

#include "reflection.hlsl"
#include "common/specularAO.hlsl"
#include "common/envBRDFApprox.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple glass shading model
use:
    - <float4> glass(<Material> material)
options:
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughness (default on mobile)
    - SCENE_BACK_SURFACE: null
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_DIRECTION: null
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
examples:
    - /shaders/lighting_raymarching_glass.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


#ifndef CAMERA_POSITION
#define CAMERA_POSITION float3(0.0, 0.0, -10.0)
#endif

#ifndef IBL_LUMINANCE
#define IBL_LUMINANCE   1.0
#endif

#ifndef FNC_PBRGLASS
#define FNC_PBRGLASS

float4 pbrGlass(const Material mat, ShadingData shadingData) {
    // Shading Data
    // ------------
#if defined(SCENE_BACK_SURFACE)
    float3 No     = normalize(mat.normal - mat.normal_back); // Normal out is the difference between the front and back normals
#else
    float3 No     = mat.normal;                            // Normal out
#endif
    float3 eta    = ior2eta(mat.ior);
    float3 f0     = float3(0.04, 0.04, 0.04);
    shadingData.N = mat.normal;
    shadingData.R = reflection(shadingData.V,  shadingData.N, mat.roughness);
    shadingData.roughness = mat.roughness; 
    shadingData.linearRoughness = mat.roughness;
    shadingData.specularColor = mat.albedo.rgb;
    shadingData.NoV = dot(No, shadingData.V);

    // Indirect Lights ( Image Based Lighting )
    // ----------------------------------------
    float3 E = envBRDFApprox(shadingData);

    float3 Gi = envMap(mat, shadingData) * E;

    #if defined(SHADING_MODEL_IRIDESCENCE)
    float3 Fr = float3(0.0, 0.0, 0.0);
    Gi  += fresnelIridescentReflection(mat.normal, -shadingData.V, f0, float3(IOR_AIR),
        mat.ior, mat.thickness, mat.roughness, Fr);
    #else
    float3 Fr = fresnel(f0, shadingData.NoV);
    Gi  += fresnelReflection(shadingData.R, Fr) * (1.0-mat.roughness);
    #endif

    float4 color  = float4(0.0, 0.0, 0.0, 1.0);

    // Refraction
    color.rgb   += transparent(No, -shadingData.V, Fr, eta, mat.roughness);
    color.rgb   += Gi * IBL_LUMINANCE * mat.ambientOcclusion;

    // TODO: RaG
    //  - Add support for multiple lights
    // 
    {
        #if defined(LIGHT_DIRECTION)
        LightDirectional L = LightDirectionalNew();
        #elif defined(LIGHT_POSITION)
        LightPoint L = LightPointNew();
        #endif

        #if defined(LIGHT_DIRECTION) || defined(LIGHT_POSITION)

        shadingData.L = L.direction;
        shadingData.H = normalize(L.direction + shadingData.V);
        shadingData.NoL = saturate(dot(shadingData.N, L.direction));
        shadingData.NoH = saturate(dot(shadingData.N, shadingData.H));
        float3 spec = specular(shadingData);

        color.rgb += L.color * spec;

        #endif
    }

    return color;
}

float4 pbrGlass(const in Material mat) {
    ShadingData shadingData = shadingDataNew();
    shadingData.V = normalize(CAMERA_POSITION - mat.position);
    return pbrGlass(mat, shadingData);
}

#endif
