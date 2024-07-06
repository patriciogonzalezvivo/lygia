#include "material.hlsl"
#include "fresnelReflection.hlsl"
#include "light/point.hlsl"
#include "light/new.hlsl"
#include "light/resolve.hlsl"
#include "light/directional.hlsl"
#include "envMap.hlsl"

#include "ior/2f0.hlsl"

#include "reflection.hlsl"
#include "common/ggx.hlsl"
#include "common/kelemen.hlsl"
#include "common/specularAO.hlsl"
#include "common/envBRDFApprox.hlsl"
/*
contributors: Patricio Gonzalez Vivo
description: Simple PBR shading model
use: <float4> pbr( <Material> _material )
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
    - LIGHT_POSITION: in hlslViewer is u_light
    - LIGHT_COLOR in hlslViewer is u_lightColor
    - CAMERA_POSITION: in hlslViewer is u_camera
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef CAMERA_POSITION
#if defined(UNITY_COMPILER_HLSL)
#define CAMERA_POSITION _WorldSpaceCameraPos
#else
#define CAMERA_POSITION float3(0.0, 0.0, -10.0)
#endif
#endif

#ifndef LIGHT_POSITION
#if defined(UNITY_COMPILER_HLSL)
#define LIGHT_POSITION _WorldSpaceLightPos0.xyz
#else
#define LIGHT_POSITION  float3(0.0, 10.0, -50.0)
#endif
#endif

#ifndef LIGHT_COLOR
#if defined(UNITY_COMPILER_HLSL)
#include <UnityLightingCommon.cginc>
#define LIGHT_COLOR     _LightColor0.rgb
#else
#define LIGHT_COLOR     float3(0.5, 0.5, 0.5)
#endif
#endif

#ifndef LIGHT_INTENSITY
#define LIGHT_INTENSITY 1.0
#endif

#ifndef IBL_LUMINANCE
#define IBL_LUMINANCE   1.0
#endif

#ifndef FNC_PBRCLEARCOAT
#define FNC_PBRCLEARCOAT

float4 pbrClearCoat(const Material _mat)
{
    // Calculate Color
    float3 diffuseColor = _mat.albedo.rgb * (float3(1.0, 1.0, 1.0) - _mat.f0) * (1.0 - _mat.metallic);
    float3 specularColor = lerp(_mat.f0, _mat.albedo.rgb, _mat.metallic);

    // Cached
    Material M = _mat;
    M.V = normalize(CAMERA_POSITION - M.position); // View
    M.NoV = dot(M.normal, M.V); // Normal . View
    M.R = reflection(M.V, M.normal, M.roughness); // Reflection

    float3 f0 = ior2f0(M.ior);
    float3 R = reflection(M.V, M.normal, M.roughness);

    #if defined(MATERIAL_HAS_NORMAL) || defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    // We want to use the geometric normal for the clear coat layer
    float clearCoatNoV      = clampNoV(dot(M.clearCoatNormal, M.V));
    float3 clearCoatNormal    = M.clearCoatNormal;
    #else
    float clearCoatNoV = M.NoV;
    float3 clearCoatNormal = M.normal;
    #endif

    // Ambient Occlusion
    // ------------------------
    float ssao = 1.0;
// #if defined(FNC_SSAO) && defined(SCENE_DEPTH) && defined(RESOLUTION) && defined(CAMERA_NEAR_CLIP) && defined(CAMERA_FAR_CLIP)
//     float2 pixel = 1.0/RESOLUTION;
//     ssao = ssao(SCENE_DEPTH, gl_FragCoord.xy*pixel, pixel, 1.);
// #endif 

    // Global Ilumination ( mage Based Lighting )
    // ------------------------
    float3 E = envBRDFApprox(specularColor, M);

    // // This is a bit of a hack to pop the metalics
    // float specIntensity =   (2.0 * M.metallic) * 
    //                         saturate(-1.1 + NoV + M.metallic) *          // Fresnel
    //                         (M.metallic + (.95 - M.roughness) * 2.0); // make smaller highlights brighter

    float diffAO = min(M.ambientOcclusion, ssao);
    float specAO = specularAO(M, diffAO);

    float3 Fr = float3(0.0, 0.0, 0.0);
    Fr = envMap(M) * E;
    #if !defined(PLATFORM_RPI)
    Fr += fresnelReflection(M);
    #endif
    Fr *= specAO;

    float3 Fd = diffuseColor;
    #if defined(SCENE_SH_ARRAY)
    Fd *= tonemap( sphericalHarmonics(M.normal) );
    #endif
    Fd *= diffAO;
    Fd *= (1.0 - E);

    float3 Fc = fresnel(f0, clearCoatNoV) * M.clearCoat;
    float3 attenuation = 1.0 - Fc;
    Fd *= attenuation;
    Fr *= attenuation;

    // float3 clearCoatLobe = isEvaluateSpecularIBL(p, clearCoatNormal, V, clearCoatNoV);
    float3 clearCoatR = reflection(M.V, clearCoatNormal, M.clearCoatRoughness);
    float3 clearCoatE = envBRDFApprox(f0, clearCoatNoV, M.clearCoatRoughness);
    float3 clearCoatLobe = float3(0.0, 0.0, 0.0);
    clearCoatLobe += envMap(clearCoatR, M.clearCoatRoughness, 1.0) * clearCoatE * 3.;
    clearCoatLobe += tonemap(fresnelReflection(clearCoatR, f0, clearCoatNoV)) * (1.0 - M.clearCoatRoughness) * 0.2;
    Fr += clearCoatLobe * (specAO * M.clearCoat);

    float4 color = float4(0.0, 0.0, 0.0, 1.0);
    color.rgb += Fd * IBL_LUMINANCE; // Diffuse
    color.rgb += Fr * IBL_LUMINANCE; // Specular

    // LOCAL ILUMINATION
    // ------------------------
    float3 lightDiffuse = float3(0.0, 0.0, 0.0);
    float3 lightSpecular = float3(0.0, 0.0, 0.0);
    
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
        lightResolve(diffuseColor, specularColor, M, L, lightDiffuse, lightSpecular);

        color.rgb += lightDiffuse; // Diffuse
        color.rgb += lightSpecular; // Specular

        float3 h = normalize(M.V + L.direction);
        float NoH = saturate(dot(M.normal, h));
        float NoL = saturate(dot(M.normal, L.direction));
        float LoH = saturate(dot(L.direction, h));

        #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
        // If the material has a normal map, we want to use the geometric normal
        // instead to avoid applying the normal map details to the clear coat layer
        N = clearCoatNormal;
        float clearCoatNoH = saturate(dot(clearCoatNormal, h));
        #else
        float clearCoatNoH = saturate(dot(M.normal, M.V));
        #endif

        // clear coat specular lobe
        float D = GGX(M.normal, h, clearCoatNoH, M.clearCoatRoughness);
        float3 F = fresnel(f0, LoH) * M.clearCoat;

        float3 Fcc = F;
        float3 clearCoat = float3(D, D, D) * kelemen(LoH); // * F;
        float3 atten = (1.0 - Fcc);

        #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
        // If the material has a normal map, we want to use the geometric normal
        // instead to avoid applying the normal map details to the clear coat layer
        float clearCoatNoL = saturate(dot(clearCoatNormal, L.direction));
        color.rgb = color.rgb * atten * NoL + (clearCoat * clearCoatNoL * L.color) * L.intensity;// * L.shadow;
        #else
        color.rgb = color.rgb * atten + (clearCoat * L.color) * (L.intensity * NoL); //(L.intensity * L.shadow * NoL);
        #endif

        #endif
    }
    
    // Final
    color.rgb *= M.ambientOcclusion;
    color.rgb += M.emissive;
    color.a = M.albedo.a;

    return color;
}
#endif
