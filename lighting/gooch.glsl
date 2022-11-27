#include "material/roughness.glsl"
#include "material/normal.glsl"
#include "material/albedo.glsl"

#include "diffuse.glsl"
#include "specular.glsl"

#include "material.glsl"

#include "../sample/shadowPCF.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: render with a gooch stylistic shading model
use: <vec4> gooch(<vec4> albedo, <vec3> normal, <vec3> light, <vec3> view, <float> roughness)
options:
    - GOOCH_WARM: defualt vec3(0.25, 0.15, 0.0)
    - GOOCH_COLD: defualt vec3(0.0, 0.0, 0.2)
    - GOOCH_SPECULAR: defualt vec3(1.0, 1.0, 1.0)
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - LIGHT_COORD:       in GlslViewer is  v_lightCoord
    - LIGHT_SHADOWMAP:   in GlslViewer is u_lightShadowMap
    - LIGHT_SHADOWMAP_SIZE: in GlslViewer is 1024.0
*/

#ifndef SURFACE_POSITION
#define SURFACE_POSITION v_position
#endif

#ifndef CAMERA_POSITION
#define CAMERA_POSITION vec3(0.0, 0.0, -10.0);
#endif

#ifndef LIGHT_POSITION
#define LIGHT_POSITION vec3(0.0, 10.0, -50.0)
#endif

#ifndef GOOCH_WARM 
#define GOOCH_WARM vec3(0.25, 0.15, 0.0)
#endif 

#ifndef GOOCH_COLD 
#define GOOCH_COLD vec3(0.0, 0.0, 0.2)
#endif 

#ifndef GOOCH_SPECULAR
#define GOOCH_SPECULAR vec3(1.0, 1.0, 1.0)
#endif 

#ifndef FNC_GOOCH
#define FNC_GOOCH
vec4 gooch(const in vec4 albedo, const in vec3 normal, const in vec3 light, const in vec3 view, const in float roughness, const in float shadow) {
    vec3 warm = GOOCH_WARM + albedo.rgb * 0.6;
    vec3 cold = GOOCH_COLD + albedo.rgb * 0.1;

    vec3 l = normalize(light);
    vec3 n = normalize(normal);
    vec3 v = normalize(view);

    // Lambert Diffuse
    float diff = diffuse(l, n, v, roughness) * shadow;
    // Phong Specular
    float spec = specular(l, n, v, roughness) * shadow;

    return vec4(mix(mix(cold, warm, diff), GOOCH_SPECULAR, spec), albedo.a);
}


vec4 gooch(const in vec4 albedo, const in vec3 normal, const in vec3 light, const in vec3 view, const in float roughness) {
    return gooch(albedo, normal, light, view, roughness, 1.0);
}

vec4 gooch(Material material) {
    vec3 pos = CAMERA_POSITION - material.position;
    #ifdef LIGHT_DIRECTION
    vec3 lig = LIGHT_DIRECTION;
    #else
    vec3 lig = LIGHT_POSITION - material.position;
    #endif
    return gooch(material.albedo, material.normal, lig, pos, material.roughness, material.shadow);
}

#endif