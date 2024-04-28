#include "../math/powFast.hlsl"
#include "../color/tonemap.hlsl"

#include "material.hlsl"
#include "fresnelReflection.hlsl"

#include "envMap.hlsl"
#include "sphericalHarmonics.hlsl"
#include "diffuse.hlsl"
#include "specular.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: simple PBR shading model
use: 
    - <float4> pbrLittle(<Material> material) 
    - <float4> pbrLittle(<float4> albedo, <float3> normal, <float> roughness, <float> metallic [, <float3> f0] ) 
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

#ifndef FNC_PBR_LITTLE
#define FNC_PBR_LITTLE

float4 pbrLittle(float4 albedo, float3 position, float3 normal, float roughness, float metallic, float3 f0, float shadow ) {
    #ifdef LIGHT_DIRECTION
    float3 L = normalize(LIGHT_DIRECTION);
    #else
    float3 L = normalize(LIGHT_POSITION - position);
    #endif
    float3 N = normalize(normal);
    float3 V = normalize(CAMERA_POSITION - position);

    float notMetal = 1. - metallic;
    float smoothness = .95 - saturate(roughness);

    // DIFFUSE
    float diff = diffuse(L, N, V, roughness) * shadow;
    float spec = specular(L, N, V, roughness) * shadow;

    albedo.rgb = albedo.rgb * diff;
    #if defined(UNITY_COMPILER_HLSL)
    albedo.rgb *= ShadeSH9(half4(N,1));
    #elif defined(SCENE_SH_ARRAY)
    albedo.rgb *= tonemapReinhard( sphericalHarmonics(N) );
    #endif

    float NoV = dot(N, V); 

    // SPECULAR
    float3 specIntensity =  float3(1.0, 1.0, 1.0) *
                            (0.04 * notMetal + 2.0 * metallic) * 
                            saturate(-1.1 + NoV + metallic) * // Fresnel
                            (metallic + smoothness * 4.0); // make smaller highlights brighter

    float3 R = reflect(-V, N);
    float3 ambientSpecular = tonemapReinhard( envMap(R, roughness, metallic) ) * specIntensity;
    ambientSpecular += fresnelReflection(R, f0, NoV) * metallic;

    albedo.rgb =    albedo.rgb * notMetal + ( ambientSpecular 
                    + LIGHT_COLOR * 2.0 * spec
                    ) * (notMetal * smoothness + albedo.rgb * metallic);

    return albedo;
}

float4 pbrLittle(float4 albedo, float3 position, float3 normal, float roughness, float metallic, float shadow) {
    return pbrLittle(albedo, position, normal, roughness, metallic, float3(0.04, 0.04, 0.04), shadow);
}

float4 pbrLittle(Material material) {
    return pbrLittle(material.albedo, material.position, material.normal, material.roughness, material.metallic, material.f0, material.ambientOcclusion * material.shadow) + float4(material.emissive, 0.0);
}

#endif