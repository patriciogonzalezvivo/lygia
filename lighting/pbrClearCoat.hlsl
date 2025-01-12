#include "../color/tonemap.hlsl"

#include "common/ggx.hlsl"
#include "common/kelemen.hlsl"

#include "shadingData/new.hlsl"
#include "material.hlsl"
#include "envMap.hlsl"
#include "fresnelReflection.hlsl"
#include "sphericalHarmonics.hlsl"
#include "light/new.hlsl"
#include "light/resolve.hlsl"

#include "reflection.hlsl"
#include "common/specularAO.hlsl"
#include "common/envBRDFApprox.hlsl"

#include "ior/2f0.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple PBR shading model
use: <float4> pbr( <Material> material )
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

#ifndef LIGHT_INTENSITY
#define LIGHT_INTENSITY 1.0
#endif

#ifndef IBL_LUMINANCE
#define IBL_LUMINANCE   1.0
#endif

#ifndef FNC_PBRCLEARCOAT
#define FNC_PBRCLEARCOAT

float4 pbrClearCoat(const Material mat, ShadingData shadingData) {
    shadingDataNew(mat, shadingData);

    float3    f0      = ior2f0(mat.ior);
    float3    R       = reflection(shadingData.V, mat.normal, mat.roughness);

    #if defined(MATERIAL_HAS_NORMAL) || defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    // We want to use the geometric normal for the clear coat layer
    float clearCoatNoV      = clampNoV(dot(mat.clearCoatNormal, shadingData.V));
    float3 clearCoatNormal    = mat.clearCoatNormal;
    #else
    float clearCoatNoV      = shadingData.NoV;
    float3 clearCoatNormal    = mat.normal;
    #endif

    // Indirect Lights ( Image Based Lighting )
    // ----------------------------------------
    float3 E = envBRDFApprox(shadingData);

    float diffAO = mat.ambientOcclusion;
    float specAO = specularAO(mat, shadingData, diffAO);

    float3 Fr = float3(0.0, 0.0, 0.0);
    Fr = envMap(mat, shadingData) * E;
    #if !defined(PLATFORM_RPI)
    Fr  += fresnelReflection(mat, shadingData);
    #endif
    Fr *= specAO;

    float3 Fd = shadingData.diffuseColor;
    #if defined(SCENE_SH_ARRAY)
    Fd *= tonemap( sphericalHarmonics(mat.normal) );
    #endif
    Fd *= diffAO;
    Fd *= (1.0 - E);

    float3 Fc = fresnel(f0, clearCoatNoV) * mat.clearCoat;
    float3 attenuation = 1.0 - Fc;
    Fd *= attenuation;
    Fr *= attenuation;

    // float3 clearCoatLobe = isEvaluateSpecularIBL(p, clearCoatNormal, V, clearCoatNoV);
    float3 clearCoatR = reflection(shadingData.V, clearCoatNormal, mat.clearCoatRoughness);
    float3 clearCoatE = envBRDFApprox(f0, clearCoatNoV, mat.clearCoatRoughness);
    float3 clearCoatLobe = float3(0.0, 0.0, 0.0);
    clearCoatLobe += envMap(clearCoatR, mat.clearCoatRoughness, 1.0) * clearCoatE * 3.;
    clearCoatLobe += tonemap( fresnelReflection(clearCoatR, f0, clearCoatNoV) ) * (1.0-mat.clearCoatRoughness) * 0.2;
    Fr += clearCoatLobe * (specAO * mat.clearCoat);

    float4 color  = float4(0.0, 0.0, 0.0, 1.0);
    color.rgb  += Fd * IBL_LUMINANCE;    // Diffuse
    color.rgb  += Fr * IBL_LUMINANCE;    // Specular

    // Direct Lights
    // -------------
    // TODO: 
    //  - Add support for multiple lights
    // 
    {
        #if defined(LIGHT_DIRECTION)
        LightDirectional L = LightDirectionalNew();
        #elif defined(LIGHT_POSITION)
        LightPoint L = LightPointNew();
        #endif

        #if defined(LIGHT_DIRECTION) || defined(LIGHT_POSITION)
        lightResolve(L, mat, shadingData);

        color.rgb  += shadingData.directDiffuse;     // Diffuse
        color.rgb  += shadingData.directSpecular;    // Specular

        float3  h     = normalize(shadingData.V + L.direction);
        float NoH   = saturate(dot(mat.normal, h));
        float NoL   = saturate(dot(mat.normal, L.direction));
        float LoH   = saturate(dot(L.direction, h));

        #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
        // If the material has a normal map, we want to use the geometric normal
        // instead to avoid applying the normal map details to the clear coat layer
        N = clearCoatNormal;
        float clearCoatNoH = saturate(dot(clearCoatNormal, h));
        #else
        float clearCoatNoH = saturate(dot(mat.normal, shadingData.V));
        #endif

        // clear coat specular lobe
        float D         =   GGX(mat.normal, h, clearCoatNoH, mat.clearCoatRoughness);
        float3  F         =   fresnel(f0, LoH) * mat.clearCoat;

        float3  Fcc       =   F;
        float3  clearCoat =   float3(D, D, D) * kelemen(LoH);// * F;
        float3  atten     =   (1.0 - Fcc);

        #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
        // If the material has a normal map, we want to use the geometric normal
        // instead to avoid applying the normal map details to the clear coat layer
        float clearCoatNoL = saturate(dot(clearCoatNormal, L.direction));
        color.rgb = color.rgb * atten * NoL + (clearCoat * clearCoatNoL * L.color) * L.intensity;// * L.shadow;
        #else
        color.rgb = color.rgb * atten + (clearCoat * L.color) * (L.intensity * NoL);//(L.intensity * L.shadow * NoL);
        #endif

        #endif
    }
    
    // Final
    color.rgb  *= mat.ambientOcclusion;
    color.rgb  += mat.emissive;
    color.a     = mat.albedo.a;

    return color;
}

float4 pbrClearCoat(const in Material mat) {
    ShadingData shadingData = shadingDataNew();
    shadingData.V = normalize(CAMERA_POSITION - mat.position);
    return pbrClearCoat(mat, shadingData);
}
#endif