#include "../math/powFast.glsl"
#include "../math/saturate.glsl"
#include "../color/tonemap.glsl"

#include "shadow.glsl"
#include "material.glsl"
#include "fresnelReflection.glsl"
#include "sphericalHarmonics.glsl"

#include "ior.glsl"
#include "envMap.glsl"
#include "diffuse.glsl"
#include "specular.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple PBR shading model
use:
    - <vec4> pbrLittle(<Material> material)
    - <vec4> pbrLittle(<vec4> albedo, <vec3> normal, <float> roughness, <float> metallic [, <vec3> f0] )
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
#define CAMERA_POSITION vec3(0.0, 0.0, -10.0)
#endif

#ifndef LIGHT_POSITION
#define LIGHT_POSITION  vec3(0.0, 10.0, -50.0)
#endif

#ifndef LIGHT_COLOR
#define LIGHT_COLOR     vec3(0.5, 0.5, 0.5)
#endif

#ifndef FNC_PBR_LITTLE
#define FNC_PBR_LITTLE

vec4 pbrLittle(Material mat, ShadingData shadingData) {
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
    float shadow = shadow(LIGHT_SHADOWMAP, vec2(LIGHT_SHADOWMAP_SIZE), (LIGHT_COORD).xy, (LIGHT_COORD).z);
    #elif defined(FNC_RAYMARCH_SOFTSHADOW)
    float shadow = raymarchSoftShadow(mat.position, shadingData.L);
    #else
    float shadow = 1.0;
    #endif

    // DIFFUSE
    float diff = diffuse(shadingData) * shadow;
    vec3 spec = specular(shadingData) * shadow;

    vec3 albedo = mat.albedo.rgb * diff;
// #ifdef SCENE_SH_ARRAY
//     albedo.rgb += tonemap( sphericalHarmonics(shadingData.N) ) * 0.25;
// #endif

    // SPECULAR
    // This is a bit of a stylistic approach
    float specIntensity =   (0.04 * notMetal + 2.0 * mat.metallic) * 
                            saturate(-1.1 + shadingData.NoV + mat.metallic) * // Fresnel
                            (mat.metallic + smoothness * 4.0); // make smaller highlights brighter

    vec3 ambientSpecular = tonemap( envMap(mat, shadingData) ) * specIntensity;
    ambientSpecular += fresnelReflection(mat, shadingData) * (1.0-mat.roughness);

    albedo =    albedo.rgb * notMetal + ( ambientSpecular 
                + LIGHT_COLOR * 2.0 * spec
                ) * (notMetal * smoothness + albedo * mat.metallic);

    return vec4(albedo, mat.albedo.a);
}

vec4 pbrLittle(const in Material mat) {
    ShadingData shadingData = shadingDataNew();
    shadingData.V = normalize(CAMERA_POSITION - mat.position);
    return pbrLittle(mat, shadingData);
}

#endif