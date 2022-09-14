#include "material/roughness.glsl"
#include "material/normal.glsl"
#include "material/baseColor.glsl"

#include "diffuse.glsl"
#include "specular.glsl"

#include "material.glsl"

#include "../sample/shadowPCF.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: render with a gooch stylistic shading model
use: <vec4> gooch(<vec4> baseColor, <vec3> normal, <vec3> light, <vec3> view, <float> roughness)
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
#if defined(GLSLVIEWER)
#define CAMERA_POSITION u_camera
#else
#define CAMERA_POSITION vec3(0.0, 0.0, -10.0);
#endif
#endif


#ifndef LIGHT_POSITION
#if defined(GLSLVIEWER)
#define LIGHT_POSITION u_light
#else
#define LIGHT_POSITION vec3(0.0, 10.0, -50.0)
#endif
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
vec4 gooch(vec4 baseColor, vec3 normal, vec3 light, vec3 view, float roughness) {
    vec3 warm = GOOCH_WARM + baseColor.rgb * 0.6;
    vec3 cold = GOOCH_COLD + baseColor.rgb * 0.1;

    vec3 l = normalize(light);
    vec3 n = normalize(normal);
    vec3 v = normalize(view);

    // Lambert Diffuse
    float diffuse = diffuse(l, n, v, roughness);
    // Phong Specular
    float specular = specular(l, n, v, roughness);

#if defined(LIGHT_SHADOWMAP) && defined(LIGHT_SHADOWMAP_SIZE) && defined(LIGHT_COORD)
    float bias = 0.005;
    float shadow = sampleShadowPCF(u_lightShadowMap, vec2(LIGHT_SHADOWMAP_SIZE), (LIGHT_COORD).xy, (LIGHT_COORD).z - bias);
    specular *= shadow;
    diffuse *= shadow;
#endif

    return vec4(mix(mix(cold, warm, diffuse), GOOCH_SPECULAR, specular), baseColor.a);
}

vec4 gooch(Material material) {
    return gooch(material.baseColor, material.normal, LIGHT_POSITION, (CAMERA_POSITION - material.position), material.roughness);
}

#endif