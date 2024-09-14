#include "shadingData/new.glsl"
#include "material/roughness.glsl"
#include "material/normal.glsl"
#include "material/albedo.glsl"
#include "material.glsl"
#include "light/new.glsl"
#include "specular.glsl"
#include "diffuse.glsl"
#include "reflection.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Render with a gooch stylistic shading model
use: <vec4> gooch(<vec4> albedo, <vec3> normal, <vec3> light, <vec3> view, <float> roughness)
options:
    - GOOCH_WARM: defualt vec3(0.25, 0.15, 0.0)
    - GOOCH_COLD: defualt vec3(0.0, 0.0, 0.2)
    - GOOCH_SPECULAR: defualt vec3(1.0, 1.0, 1.0)
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - LIGHT_COORD: in GlslViewer is  v_lightCoord
    - LIGHT_SHADOWMAP: in GlslViewer is u_lightShadowMap
    - LIGHT_SHADOWMAP_SIZE: in GlslViewer is 1024.0
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef CAMERA_POSITION
#define CAMERA_POSITION vec3(0.0, 0.0, -10.0)
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

    ShadingData shadingData = shadingDataNew();
    shadingData.L = normalize(_L);
    shadingData.N = normalize(_N);
    shadingData.V = normalize(_V);
    shadingData.H = normalize(shadingData.L + shadingData.V);
    shadingData.NoV = dot(shadingData.N, shadingData.V);
    shadingData.NoL = dot(shadingData.N, shadingData.L);
    shadingData.NoH = saturate(dot(shadingData.N, shadingData.H));
    shadingData.roughness = _roughness;

    // Lambert Diffuse
    float diff = diffuse(shadingData) * _Li;
    // Phong Specular
    vec3 spec = vec3(1.0, 1.0, 1.0) * specularBlinnPhongRoughness(shadingData) * _Li;

    return vec4(mix(mix(cold, warm, diff), GOOCH_SPECULAR, spec), _albedo.a);
}

vec4 gooch(const in LightDirectional _L, in Material _M, ShadingData shadingData) {
    return gooch(_M.albedo, _M.normal, _L.direction, shadingData.V, _M.roughness, _L.intensity);
}

vec4 gooch(const in LightPoint _L, in Material _M, ShadingData shadingData) {
    return gooch(_M.albedo, _M.normal, _L.position, shadingData.V, _M.roughness, _L.intensity);
}

vec4 gooch(const in Material _M, ShadingData shadingData) {
    #if defined(LIGHT_DIRECTION)
    LightDirectional L;
    #elif defined(LIGHT_POSITION)
    LightPoint L;
    #endif
    lightNew(L);

    #if defined(FNC_RAYMARCH_SOFTSHADOW)
    #if defined(LIGHT_DIRECTION)
    L.intensity *= raymarchSoftShadow(_M.position, L.direction);
    #elif defined(LIGHT_POSITION)
    L.intensity *= raymarchSoftShadow(_M.position, L.position);
    #endif
    #endif 

    return gooch(L, _M, shadingData) * _M.ambientOcclusion;
}

vec4 gooch(const in Material _M) {
    ShadingData shadingData = shadingDataNew();
    shadingData.V = normalize(CAMERA_POSITION - _M.position);
    return gooch(_M, shadingData);
}

#endif