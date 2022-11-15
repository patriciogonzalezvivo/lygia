#include "../math/powFast.glsl"
#include "../math/saturate.glsl"
#include "../color/tonemap.glsl"

#include "shadow.glsl"
#include "material.glsl"
#include "fresnelReflection.glsl"

#include "envMap.glsl"
#include "sphericalHarmonics.glsl"
#include "diffuse.glsl"
#include "specular.glsl"

#include "../math/saturate.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: simple PBR shading model
use: 
    - <vec4> pbrLittle(<Material> material) 
    - <vec4> pbrLittle(<vec4> albedo, <vec3> normal, <float> roughness, <float> metallic [, <vec3> f0] ) 
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
*/

#ifndef CAMERA_POSITION
#define CAMERA_POSITION vec3(0.0, 0.0, -10.0);
#endif

#ifndef LIGHT_POSITION
#define LIGHT_POSITION  vec3(0.0, 10.0, -50.0)
#endif

#ifndef LIGHT_COLOR
#define LIGHT_COLOR     vec3(0.5)
#endif

#ifndef FNC_PBR_LITTLE
#define FNC_PBR_LITTLE

vec4 pbrLittle(vec4 albedo, vec3 position, vec3 normal, float roughness, float metallic, vec3 f0, float shadow ) {
    #ifdef LIGHT_DIRECTION
    vec3 L = normalize(LIGHT_DIRECTION);
    #else
    vec3 L = normalize(LIGHT_POSITION - position);
    #endif
    vec3 N = normalize(normal);
    vec3 V = normalize(CAMERA_POSITION - position);

    float notMetal = 1. - metallic;
    float smooth = .95 - saturate(roughness);

    // DIFFUSE
    float diff = diffuse(L, N, V, roughness) * shadow;
    float spec = specular(L, N, V, roughness) * shadow;

    albedo.rgb = albedo.rgb * diff;
#ifdef SCENE_SH_ARRAY
    albedo.rgb *= tonemapReinhard( sphericalHarmonics(N) );
#endif

    float NoV = dot(N, V); 

    // SPECULAR
    // This is a bit of a stilistic proach
    float specIntensity =   (0.04 * notMetal + 2.0 * metallic) * 
                            saturate(-1.1 + NoV + metallic) * // Fresnel
                            (metallic + smooth * 4.0); // make smaller highlights brighter

    vec3 R = reflect(-V, N);
    vec3 ambientSpecular = tonemapReinhard( envMap(R, roughness, metallic) ) * specIntensity;
    ambientSpecular += fresnelReflection(R, f0, NoV) * metallic;

    albedo.rgb = albedo.rgb * notMetal + ( ambientSpecular 
                    + LIGHT_COLOR * 2.0 * spec
                    ) * (notMetal * smooth + albedo.rgb * metallic);

    return albedo;
}

vec4 pbrLittle(vec4 albedo, vec3 position, vec3 normal, float roughness, float metallic, float shadow) {
    return pbrLittle(albedo, position, normal, roughness, metallic, vec3(0.04), shadow);
}

vec4 pbrLittle(vec4 albedo, vec3 position, vec3 normal, float roughness, float metallic) {
    return pbrLittle(albedo, position, normal, roughness, metallic, vec3(0.04), 1.0);
}

vec4 pbrLittle(vec4 albedo, vec3 normal, float roughness, float metallic) {
    return pbrLittle(albedo, vec3(0.0), normal, roughness, metallic, vec3(0.04), 1.0);
}

vec4 pbrLittle(Material material) {
    return pbrLittle(material.albedo, material.position, material.normal, material.roughness, material.metallic, material.f0, material.ambientOcclusion * material.shadow) + vec4(material.emissive, 0.0);
}

#endif