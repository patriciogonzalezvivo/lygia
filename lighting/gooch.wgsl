#include "shadingData/new.wgsl"
#include "material/roughness.wgsl"
#include "material/normal.wgsl"
#include "material/albedo.wgsl"
#include "material.wgsl"
#include "light/new.wgsl"
#include "specular.wgsl"
#include "diffuse.wgsl"
#include "reflection.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Render with a gooch stylistic shading model
use: <vec4> gooch(<vec4> albedo, <vec3> normal, <vec3> light, <vec3> view, <float> roughness)
options:
    - GOOCH_WARM: default vec3(0.25, 0.15, 0.0)
    - GOOCH_COLD: default vec3(0.0, 0.0, 0.2)
    - GOOCH_SPECULAR: default vec3(1.0, 1.0, 1.0)
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - LIGHT_COORD: in GlslViewer is  v_lightCoord
    - LIGHT_SHADOWMAP: in GlslViewer is u_lightShadowMap
    - LIGHT_SHADOWMAP_SIZE: in GlslViewer is 1024.0
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define CAMERA_POSITION vec3(0.0, 0.0, -10.0)

// #define LIGHT_POSITION vec3(0.0, 10.0, -50.0)

// #define GOOCH_WARM vec3(0.25, 0.15, 0.0)

// #define GOOCH_COLD vec3(0.0, 0.0, 0.2)

// #define GOOCH_SPECULAR vec3(1.0, 1.0, 1.0)

fn gooch4(_albedo: vec4f, _N: vec3f, _L: vec3f, _V: vec3f, _roughness: f32, _Li: f32) -> vec4f {
    let warm = GOOCH_WARM + _albedo.rgb * 0.6;
    let cold = GOOCH_COLD + _albedo.rgb * 0.1;

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
    let diff = diffuse(shadingData) * _Li;
    // Phong Specular
    let spec = vec3f(1.0, 1.0, 1.0) * specularBlinnPhongRoughness(shadingData) * _Li;

    return vec4f(mix(mix(cold, warm, diff), GOOCH_SPECULAR, spec), _albedo.a);
}

fn gooch(_L: LightDirectional, _M: Material, shadingData: ShadingData) -> vec4f {
    return gooch(_M.albedo, _M.normal, _L.direction, shadingData.V, _M.roughness, _L.intensity);
}

fn goocha(_L: LightPoint, _M: Material, shadingData: ShadingData) -> vec4f {
    return gooch(_M.albedo, _M.normal, _L.position, shadingData.V, _M.roughness, _L.intensity);
}

fn goochb(_M: Material, shadingData: ShadingData) -> vec4f {
    LightDirectional L;
    LightPoint L;
    lightNew(L);

    L.intensity *= raymarchSoftShadow(_M.position, L.direction);
    L.intensity *= raymarchSoftShadow(_M.position, L.position);

    return gooch(L, _M, shadingData) * _M.ambientOcclusion;
}

fn goochc(_M: Material) -> vec4f {
    ShadingData shadingData = shadingDataNew();
    shadingData.V = normalize(CAMERA_POSITION - _M.position);
    return gooch(_M, shadingData);
}
