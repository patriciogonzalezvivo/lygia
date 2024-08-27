#include "../math/powFast.glsl"
#include "../math/saturate.glsl"
#include "../color/tonemap/reinhard.glsl"

#include "shadow.glsl"
#include "material.glsl"
#include "fresnelReflection.glsl"
#include "sphericalHarmonics.glsl"

#include "ior.glsl"
#include "envMap.glsl"
#include "diffuse.glsl"
#include "specular.glsl"

#include "../math/saturate.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple PBR shading model
use:
    - <vec4> pbrLittle(<Material> material)
    - <vec4> pbrLittle(<vec4> albedo, <vec3> normal, <float> roughness, <float> metallic [, <vec3> f0] )
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
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
#define LIGHT_COLOR     vec3(0.5)
#endif

#ifndef FNC_PBR_LITTLE
#define FNC_PBR_LITTLE

vec4 pbrLittle(Material mat, ShadingData shadingData) {
    shadingData.N = mat.normal;
    shadingData.R = reflect(-shadingData.V,  shadingData.N);
    shadingData.fresnel = max(mat.f0.r, max(mat.f0.g, mat.f0.b));
    shadingData.roughness = mat.roughness;
    shadingData.linearRoughness = mat.roughness;
    shadingData.diffuseColor = mat.albedo.rgb * (vec3(1.0) - mat.f0) * (1.0 - mat.metallic);
    shadingData.specularColor = mix(mat.f0, mat.albedo.rgb, mat.metallic);
    shadingData.NoV = dot(shadingData.N, shadingData.V);
    #ifdef LIGHT_DIRECTION
    shadingData.L = normalize(LIGHT_DIRECTION);
    #else
    shadingData.L = normalize(LIGHT_POSITION - mat.position);
    #endif
    shadingData.NoL = dot(shadingData.N, shadingData.L);

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
    float spec = specular(shadingData) * shadow;

    vec3 albedo = mat.albedo.rgb * diff;
// #ifdef SCENE_SH_ARRAY
    // _albedo.rgb = _albedo.rgb + tonemapReinhard( sphericalHarmonics(N) ) * 0.25;
// #endif

    // SPECULAR
    // This is a bit of a stylistic approach
    float specIntensity =   (0.04 * notMetal + 2.0 * mat.metallic) * 
                            saturate(-1.1 + shadingData.NoV + mat.metallic) * // Fresnel
                            (mat.metallic + smoothness * 4.0); // make smaller highlights brighter

    vec3 ambientSpecular = tonemapReinhard( envMap(mat, shadingData) ) * specIntensity;
    ambientSpecular += fresnelReflection(mat, shadingData) * (1.0-mat.roughness);

    albedo = albedo.rgb * notMetal + ( ambientSpecular 
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