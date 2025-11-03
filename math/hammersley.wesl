import lygia::math::consts::PI;

/*
contributors: Holger Dammertz
description: Return a Hammersley point
use: hammersley(index: u32, numSamples: i32) -> vec2f
license: CC BY 3.0 Copyright (c) 2012 Holger Dammertz
*/

fn hammersley(index: u32, numSamples: i32) -> vec2f {
    const tof: f32 = 0.5 / f32(0x80000000u);
    var bits = index;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return vec2(f32(index) / f32(numSamples), f32(bits) * tof);
}

fn hemisphereCosSample(u: vec2f) -> vec3f {
    let phi = 2.0 * PI * u.x;
    let cosTheta2 = 1.0 - u.y;
    let cosTheta = sqrt(cosTheta2);
    let sinTheta = sqrt(1.0 - cosTheta2);
    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}
