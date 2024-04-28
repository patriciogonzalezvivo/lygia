#include "../math/powFast.glsl"
#include "../math/saturate.glsl"
#include "../color/tonemap/reinhard.glsl"

#include "shadow.glsl"
#include "material.glsl"
#include "fresnelReflection.glsl"

#include "ior.glsl"
#include "envMap.glsl"
#include "diffuse.glsl"
#include "specular.glsl"

#include "../math/saturate.glsl"

/*
contributors: Patricio Gonzalez Vivo
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

vec4 pbrLittle( vec4 _albedo, vec3 _position, vec3 _normal, float _roughness, float _metallic, vec3 _f0, // Material Basic
                vec3 ior, float thickness,                                                               // Material Iridescence
                float shadow  ) {                                                                        // Light       
            
    #ifdef LIGHT_DIRECTION
    vec3 L = normalize(LIGHT_DIRECTION);
    #else
    vec3 L = normalize(LIGHT_POSITION - _position);
    #endif
    vec3 N = normalize(_normal);
    vec3 V = normalize(CAMERA_POSITION - _position);

    float notMetal = 1. - _metallic;
    float smoothness = .95 - saturate(_roughness);

    // DIFFUSE
    float diff = diffuse(L, N, V, _roughness) * shadow;
    float spec = specular(L, N, V, _roughness) * shadow;

    _albedo.rgb = _albedo.rgb * diff;
#ifdef SCENE_SH_ARRAY
    _albedo.rgb *= tonemapReinhard( sphericalHarmonics(N) );
#endif

    float NoV = dot(N, V); 

    // SPECULAR
    // This is a bit of a stilistic proach
    float specIntensity =   (0.04 * notMetal + 2.0 * _metallic) * 
                            saturate(-1.1 + NoV + _metallic) * // Fresnel
                            (_metallic + smoothness * 4.0); // make smaller highlights brighter

    vec3 R = reflect(-V, N);
    vec3 ambientSpecular = tonemapReinhard( envMap(R, _roughness, _metallic) ) * specIntensity;
    ambientSpecular += fresnelReflection(R, _f0, NoV);

    _albedo.rgb = _albedo.rgb * notMetal + ( ambientSpecular 
                    + LIGHT_COLOR * 2.0 * spec
                    ) * (notMetal * smoothness + _albedo.rgb * _metallic);

    return _albedo;
}

vec4 pbrLittle(vec4 albedo, vec3 position, vec3 normal, float roughness, float metallic, vec3 f0, float shadow) {
    return pbrLittle(albedo, position, normal, roughness, metallic, f0, vec3(IOR_GLASS), 2000.0, shadow);
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
    float s = 1.0;
    #if defined(LIGHT_SHADOWMAP) && defined(LIGHT_SHADOWMAP_SIZE) && defined(LIGHT_COORD)
    s *= shadow(LIGHT_SHADOWMAP, vec2(LIGHT_SHADOWMAP_SIZE), (LIGHT_COORD).xy, (LIGHT_COORD).z);
    #endif
    return pbrLittle(material.albedo, material.position, material.normal, material.roughness, material.metallic, material.f0, material.ior, material.thickness, material.ambientOcclusion * s) + vec4(material.emissive, 0.0);
}

#endif