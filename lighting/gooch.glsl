#include "material/roughness.glsl"
#include "material/normal.glsl"
#include "material/albedo.glsl"

#include "light/new.glsl"

#include "diffuse.glsl"
#include "specular.glsl"

#include "material.glsl"

#include "../sample/shadowPCF.glsl"

/*
contributors: Patricio Gonzalez Vivo
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
vec4 gooch(const in vec4 _albedo, const in vec3 _N, const in vec3 _L, const in vec3 _V, const in float _roughness, const in float _Li) {
    vec3 warm = GOOCH_WARM + _albedo.rgb * 0.6;
    vec3 cold = GOOCH_COLD + _albedo.rgb * 0.1;

    vec3 l = normalize(_L);
    vec3 n = normalize(_N);
    vec3 v = normalize(_V);

    // Lambert Diffuse
    float diff = diffuse(l, n, v, _roughness) * _Li;
    // Phong Specular
    float spec = specular(l, n, v, _roughness) * _Li;

    return vec4(mix(mix(cold, warm, diff), GOOCH_SPECULAR, spec), _albedo.a);
}


vec4 gooch(const in vec4 _albedo, const in vec3 _N, const in vec3 _L, const in vec3 _V, const in float _roughness) {
    return gooch(_albedo, _N, _L, _V, _roughness, 1.0);
}

vec4 gooch(const in Material _M, const in LightDirectional _L) {
    vec3 V = normalize(CAMERA_POSITION - _M.position);
    return gooch(_M.albedo, _M.normal, _L.direction, V, _M.roughness, _L.intensity * _L.shadow);
}

vec4 gooch(const in Material _M, const in LightPoint _L) {
    vec3 V = normalize(CAMERA_POSITION - _M.position);
    return gooch(_M.albedo, _M.normal, _L.direction, V, _M.roughness, _L.intensity * _L.shadow);
}

vec4 gooch(const in Material _M) {
    #if defined(LIGHT_DIRECTION)
    LightDirectional L = LightDirectionalNew();
    #elif defined(LIGHT_POSITION)
    LightPoint L = LightPointNew();
    #endif

    return gooch(_M, L);
}

#endif