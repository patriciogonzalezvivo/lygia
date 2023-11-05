#include "material.hlsl"
#include "fresnelReflection.hlsl"
#include "light/point.hlsl"
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
description: simple PBR shading model
use: <float4> pbr( <Material> _material ) 
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
    - LIGHT_POSITION: in hlslViewer is u_light
    - LIGHT_COLOR in hlslViewer is u_lightColor
    - CAMERA_POSITION: in hlslViewer is u_camera
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

float4 pbrClearCoat(const Material _mat) {
    // Calculate Color
    float3    diffuseColor = _mat.albedo.rgb * (float3(1.0, 1.0, 1.0) - _mat.f0) * (1.0 - _mat.metallic);
    float3    specularColor = lerp(_mat.f0, _mat.albedo.rgb, _mat.metallic);

    float3    N     = _mat.normal;                                  // Normal
    float3    V     = normalize(CAMERA_POSITION - _mat.position);   // View
    float   NoV     = saturate(dot(N, V));                          // Normal . View
    float3    f0    = ior2f0(_mat.ior);
    float3    R     = reflection(V, N, _mat.roughness);

    #if defined(MATERIAL_HAS_NORMAL) || defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    // We want to use the geometric normal for the clear coat layer
    float clearCoatNoV      = clampNoV(dot(_mat.clearCoatNormal, V));
    float3 clearCoatNormal  = _mat.clearCoatNormal;
    #else
    float clearCoatNoV      = NoV;
    float3 clearCoatNormal  = N;
    #endif

    // Ambient Occlusion
    // ------------------------
    float ssao = 1.0;
// #if defined(FNC_SSAO) && defined(SCENE_DEPTH) && defined(RESOLUTION) && defined(CAMERA_NEAR_CLIP) && defined(CAMERA_FAR_CLIP)
//     float2 pixel = 1.0/RESOLUTION;
//     ssao = ssao(SCENE_DEPTH, gl_FragCoord.xy*pixel, pixel, 1.);
// #endif 
    float diffuseAO = min(_mat.ambientOcclusion, ssao);
    float specAO = specularAO(NoV, diffuseAO, _mat.roughness);

    // Global Ilumination ( mage Based Lighting )
    // ------------------------
    float3 E = envBRDFApprox(specularColor, NoV, _mat.roughness);

    // This is a bit of a hack to pop the metalics
    float specIntensity =   (2.0 * _mat.metallic) * 
                            saturate(-1.1 + NoV + _mat.metallic) *          // Fresnel
                            (_mat.metallic + (.95 - _mat.roughness) * 2.0); // make smaller highlights brighter


    float3 Fr = float3(0.0, 0.0, 0.0);
    Fr = tonemap( envMap(R, _mat.roughness, _mat.metallic) ) * E * specIntensity;
    Fr += tonemap( fresnelReflection(R, f0, NoV) ) * _mat.metallic * (1.0-_mat.roughness) * 0.2;
    Fr *= specAO;

    float3 Fd = float3(0.0, 0.0, 0.0);
    Fd = diffuseColor;
    #if defined(SCENE_SH_ARRAY)
    Fd *= tonemap( sphericalHarmonics(N) );
    #endif
    Fd *= diffuseAO;
    Fd *= (1.0 - E);

    float3 Fc = fresnel(f0, clearCoatNoV) * _mat.clearCoat;
    float3 attenuation = 1.0 - Fc;
    Fd *= attenuation;
    Fr *= attenuation;

    // float3 clearCoatLobe = isEvaluateSpecularIBL(p, clearCoatNormal, V, clearCoatNoV);
    float3 clearCoatR = reflection(V, clearCoatNormal, _mat.clearCoatRoughness);
    float3 clearCoatE = envBRDFApprox(f0, clearCoatNoV, _mat.clearCoatRoughness);
    float3 clearCoatLobe = float3(0.0, 0.0, 0.0);
    clearCoatLobe += tonemap( envMap(clearCoatR, _mat.clearCoatRoughness, 1.0) ) * clearCoatE;
    clearCoatLobe += tonemap( fresnelReflection(clearCoatR, f0, clearCoatNoV) ) * (1.0-_mat.clearCoatRoughness);
    Fr += clearCoatLobe * (specAO * _mat.clearCoat);

    float4 color  = float4(0.0, 0.0, 0.0, 1.0);
    color.rgb  += Fd * IBL_LUMINANCE;    // Diffuse
    color.rgb  += Fr * IBL_LUMINANCE;    // Specular

    // LOCAL ILUMINATION
    // ------------------------
    float3 lightDiffuse = float3(0.0, 0.0, 0.0);
    float3 lightSpecular = float3(0.0, 0.0, 0.0);
    
    {
        #if defined(LIGHT_DIRECTION)
        float f0 = max(_mat.f0.r, max(_mat.f0.g, _mat.f0.b));
        lightDirectional(diffuseColor, specularColor, N, V, NoV, _mat.roughness, f0, _mat.shadow, lightDiffuse, lightSpecular);
        #elif defined(LIGHT_POSITION)
        float f0 = max(_mat.f0.r, max(_mat.f0.g, _mat.f0.b));
        lightPoint(diffuseColor, specularColor, N, V, NoV, _mat.roughness , f0, _mat.shadow, lightDiffuse, lightSpecular);
        #endif
    }
    
    color.rgb  += lightDiffuse;     // Diffuse
    color.rgb  += lightSpecular;    // Specular

    // Clear Coat
    #if defined(LIGHT_DIRECTION) || defined(LIGHT_POSITION)
    #if defined(LIGHT_DIRECTION)
    float3 L = normalize(LIGHT_DIRECTION);
    #elif defined(LIGHT_POSITION)
    float3 L = normalize(LIGHT_POSITION - _mat.position);
    #endif

    float3 H = normalize(V + L);
    float NoL = saturate(dot(N, L));
    float LoH = saturate(dot(L, H));

    #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    // If the material has a normal map, we want to use the geometric normal
    // instead to avoid applying the normal map details to the clear coat layer
    N = clearCoatNormal;
    float clearCoatNoH = saturate(dot(clearCoatNormal, H));
    #else
    float clearCoatNoH = saturate(dot(N, V));
    #endif

    // clear coat specular lobe
    float D             =   GGX(N, H, clearCoatNoH, _mat.clearCoatRoughness);
    float3  F           =   fresnel(f0, LoH) * _mat.clearCoat;
    float3  Fcc         =   F;
    float3  clearCoat   =   D * 
                            kelemen(LoH) * 
                            F;
    float3  atten       =   (1.0 - Fcc);

    #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    // If the material has a normal map, we want to use the geometric normal
    // instead to avoid applying the normal map details to the clear coat layer
    float clearCoatNoL = saturate(dot(clearCoatNormal, L));
    color.rgb   = color.rgb * atten * NoL + (clearCoat * clearCoatNoL * LIGHT_COLOR) * (LIGHT_INTENSITY * _mat.shadow);
    #else
    // color.rgb = color.rgb * atten + (clearCoat * LIGHT_COLOR) * (LIGHT_INTENSITY * NoL * _mat.shadow);
    color.rgb   = color.rgb + (clearCoat * LIGHT_COLOR) * (LIGHT_INTENSITY * NoL * _mat.shadow);
    #endif

    #endif

    // Final
    color.rgb  *= _mat.ambientOcclusion;
    color.rgb  += _mat.emissive;
    color.a     = _mat.albedo.a;

    return color;
}
#endif
