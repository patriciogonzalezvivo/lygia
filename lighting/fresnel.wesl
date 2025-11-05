import lygia::lighting::common::schlick::{schlick, schlickF32};
import lygia::math::saturate::saturate;
import lygia::math::pow5::pow5;

/*
contributors: Patricio Gonzalez Vivo
description: Resolve fresnel coefficient
use:
    - <f32|vec3f> fresnel(<f32|vec3f> f0, <f32> NoV)
    - <vec3f> fresnel(<vec3f> f0, <vec3f> normal, <vec3f> view)
    - <vec3f> fresnel(<vec3f> f0, <f32> NoV, <f32> roughness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn fresnelFromVectors(f0: vec3f, normal: vec3f, view: vec3f) -> vec3f {
    return schlick(f0, 1.0, dot(view, normal));
}

fn fresnel(f0: vec3f, NoV: f32) -> vec3f {
    let f90 = saturate(dot(f0, vec3f(50.0 * 0.33)));
    return schlick(f0, f90, NoV);
}

fn fresnelF32(f0: f32, NoV: f32) -> f32 {
    return schlickF32(f0, 1.0, NoV);
}

// Roughness-adjusted fresnel function to attenuate high speculars at glancing angles
// Very useful when used with filtered environment maps
// See https://seblagarde.wordpress.com/2011/08/17/hello-world/
fn fresnelRoughness(f0: vec3f, NoV: f32, roughness: f32) -> vec3f {
    return f0 + (max(vec3f(1.0 - roughness), f0) - f0) * pow5(1.0 - NoV);
}
