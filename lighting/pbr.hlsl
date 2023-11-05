#include "material.hlsl"
#include "fresnelReflection.hlsl"

#include "light/point.hlsl"
#include "light/directional.hlsl"

#include "reflection.hlsl"
#include "common/specularAO.hlsl"
#include "common/envBRDFApprox.hlsl"
#include "envMap.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: simple PBR shading model
use: <float4> pbr( <Material> _material ) 
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
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

#ifndef IBL_LUMINANCE
#define IBL_LUMINANCE   1.0
#endif

#ifndef FNC_PBR
#define FNC_PBR

float4 pbr(const Material _mat) {
    // Calculate Color
    float3    diffuseColor = _mat.albedo.rgb * (float3(1.0, 1.0, 1.0) - _mat.f0) * (1.0 - _mat.metallic);
    float3    specularColor = lerp(_mat.f0, _mat.albedo.rgb, _mat.metallic);

    float3    N     = _mat.normal;                                  // Normal
    float3    V     = normalize(CAMERA_POSITION - _mat.position);   // View
    float   NoV     = dot(N, V);                                    // Normal . View
    float   f0      = max(_mat.f0.r, max(_mat.f0.g, _mat.f0.b));
    float roughness = _mat.roughness;
    float3    R     = reflection(V, N, roughness);

    // Ambient Occlusion
    // ------------------------
    float ssao = 1.0;
// #if defined(FNC_SSAO) && defined(SCENE_DEPTH) && defined(RESOLUTION) && defined(CAMERA_NEAR_CLIP) && defined(CAMERA_FAR_CLIP)
//     float2 pixel = 1.0/RESOLUTION;
//     ssao = ssao(SCENE_DEPTH, gl_FragCoord.xy*pixel, pixel, 1.);
// #endif 
    float diffAO = min(_mat.ambientOcclusion, ssao);
    float specAO = specularAO(NoV, diffAO, roughness);

    // Global Ilumination ( mage Based Lighting )
    // ------------------------
    float3 E = envBRDFApprox(specularColor, NoV, roughness);

    // This is a bit of a hack to pop the metalics
    float specIntensity =   (2.0 * _mat.metallic) * 
                            saturate(-1.1 + NoV + _mat.metallic) *          // Fresnel
                            (_mat.metallic + (.95 - _mat.roughness) * 2.0); // make smaller highlights brighter

    float3 Fr = float3(0.0, 0.0, 0.0);
    Fr = tonemap( envMap(R, roughness, _mat.metallic) ) * E * specIntensity;
    Fr += tonemap( fresnelReflection(R, _mat.f0, NoV) ) * _mat.metallic * (1.0-roughness) * 0.2;
    Fr *= specAO;

    float3 Fd = float3(0.0, 0.0, 0.0);
    Fd = diffuseColor;
    #if defined(UNITY_COMPILER_HLSL)
    Fd *= ShadeSH9(half4(N,1));
    #elif defined(SCENE_SH_ARRAY)
    Fd *= tonemap( sphericalHarmonics(N) );
    #endif
    Fd *= diffAO;
    Fd *= (1.0 - E);

    // Local Ilumination
    // ------------------------
    float3 lightDiffuse = float3(0.0, 0.0, 0.0);
    float3 lightSpecular = float3(0.0, 0.0, 0.0);
    
    {
        #ifdef LIGHT_DIRECTION
        lightDirectional(diffuseColor, specularColor, N, V, NoV, roughness, f0, _mat.shadow, lightDiffuse, lightSpecular);
        #elif defined(LIGHT_POSITION)
        lightPoint(diffuseColor, specularColor, N, V, NoV, roughness, f0, _mat.shadow, lightDiffuse, lightSpecular);
        #endif
    }
    
    // Final Sum
    // ------------------------
    float4 color  = float4(0.0, 0.0, 0.0, 1.0);
    color.rgb  += Fd * IBL_LUMINANCE + lightDiffuse;     // Diffuse
    color.rgb  += Fr * IBL_LUMINANCE + lightSpecular;    // Specular
    color.rgb  *= _mat.ambientOcclusion;
    color.rgb  += _mat.emissive;
    color.a     = _mat.albedo.a;

    return color;
}
#endif
