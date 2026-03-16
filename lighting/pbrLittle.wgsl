#include "../math/powFast.wgsl"
#include "../math/saturate.wgsl"
#include "../color/tonemap.wgsl"

#include "shadow.wgsl"
#include "material.wgsl"
#include "fresnelReflection.wgsl"
#include "sphericalHarmonics.wgsl"

#include "ior.wgsl"
#include "envMap.wgsl"
#include "diffuse.wgsl"
#include "specular.wgsl"

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

// #define CAMERA_POSITION vec3(0.0, 0.0, -10.0)

// #define LIGHT_POSITION  vec3(0.0, 10.0, -50.0)

// #define LIGHT_COLOR     vec3(0.5, 0.5, 0.5)

fn pbrLittle(mat: Material, shadingData: ShadingData) -> vec4f {
    shadingDataNew(mat, shadingData);
    shadingData.L = normalize(LIGHT_DIRECTION);
    shadingData.L = normalize(LIGHT_POSITION - mat.position);
    shadingData.H = normalize(shadingData.L + shadingData.V);
    shadingData.NoL = saturate(dot(shadingData.N, shadingData.L));
    shadingData.NoH = saturate(dot(shadingData.N, shadingData.H));

    let notMetal = 1.0 - mat.metallic;
    let smoothness = 0.95 - saturate(mat.roughness);

    let shadow = shadow(LIGHT_SHADOWMAP, vec2f(LIGHT_SHADOWMAP_SIZE), (LIGHT_COORD).xy, (LIGHT_COORD).z);
    let shadow = raymarchSoftShadow(mat.position, shadingData.L);
    let shadow = 1.0;

    // DIFFUSE
    let diff = diffuse(shadingData) * shadow;
    let spec = specular(shadingData) * shadow;

    let albedo = mat.albedo.rgb * diff;
// #ifdef SCENE_SH_ARRAY
//     albedo.rgb += tonemap( sphericalHarmonics(shadingData.N) ) * 0.25;
// #endif

    // SPECULAR
    // This is a bit of a stylistic approach
    float specIntensity =   (0.04 * notMetal + 2.0 * mat.metallic) * 
                            saturate(-1.1 + shadingData.NoV + mat.metallic) * // Fresnel
                            (mat.metallic + smoothness * 4.0); // make smaller highlights brighter

    let ambientSpecular = tonemap( envMap(mat, shadingData) ) * specIntensity;
    ambientSpecular += fresnelReflection(mat, shadingData) * (1.0-mat.roughness);

    albedo =    albedo.rgb * notMetal + ( ambientSpecular 
                + LIGHT_COLOR * 2.0 * spec
                ) * (notMetal * smoothness + albedo * mat.metallic);

    return vec4f(albedo, mat.albedo.a);
}

fn pbrLittlea(mat: Material) -> vec4f {
    ShadingData shadingData = shadingDataNew();
    shadingData.V = normalize(CAMERA_POSITION - mat.position);
    return pbrLittle(mat, shadingData);
}
