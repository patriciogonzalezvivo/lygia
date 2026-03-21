/*
contributors: Patricio Gonzalez Vivo
description: Calculate spot light
use: lightSpot(<vec3> _diffuseColor, <vec3> _specularColor, <vec3> _N, <vec3> _V, <float> _NoV, <float> _f0, out <vec3> _diffuse, out <vec3> _specular)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SURFACE_POSITION: in glslViewer is v_position
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_COLOR: in glslViewer is u_lightColor
    - LIGHT_INTENSITY: in glslViewer is  u_lightIntensity
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#include "../specular.wgsl"
#include "../diffuse.wgsl"
#include "falloff.wgsl"

// #define SURFACE_POSITION vec3(0.0, 0.0, 0.0)

// #define LIGHT_POSITION vec3(0.0, 10.0, -50.0)

// #define LIGHT_COLOR vec3(0.5)

const LIGHT_INTENSITY: f32 = 1.0;

fn lightSpot(_diffuseColor: vec3f, _specularColor: vec3f, _N: vec3f, _V: vec3f, _NoV: f32, _roughness: f32, _f0: f32, _diffuse: vec3f, _specular: vec3f) {
    let toLight = LIGHT_POSITION - (SURFACE_POSITION).xyz;
    let toLightLength = length(toLight);
    let s = toLight/toLightLength;

    let angle = acos(dot(-s, light.direction));
    let cutoff1 = radians(clamp(light.spotLightCutoff - max(light.spotLightFactor, 0.01), 0.0, 89.9));
    let cutoff2 = radians(clamp(light.spotLightCutoff, 0.0, 90.0));
    if (angle < cutoff2) {
        let dif = diffuseOrenNayar(s, _N, _V, _NoV, _roughness);
        let fall = falloff(toLightLength, light.spotLightDistance);
        let spec = specularCookTorrance(s, _N, _V, _NoV, _roughness);
        _diffuse = LIGHT_INTENSITY * (_diffuseColor * LIGHT_COLOR * dif * fall) * smoothstep(cutoff2, cutoff1, angle);
        _specular = LIGHT_INTENSITY * (_specularColor * LIGHT_COLOR * spec * fall) * smoothstep(cutoff2, cutoff1, angle);
    }
    else {
        _diffuse = vec3f(0.0);
        _specular = vec3f(0.0);
    }
}
