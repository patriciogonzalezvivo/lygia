/*
contributors: Narkowicz 2015
description: ACES Filmic Tone Mapping Curve. https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
use: tonemapACES(<vec3f|vec4f> x)
*/

import lygia::math::saturate::saturate3;

const aces_a: f32 = 2.51;
const aces_b: f32 = 0.03;
const aces_c: f32 = 2.43;
const aces_d: f32 = 0.59;
const aces_e: f32 = 0.14;

fn tonemapACES3(v: vec3f) -> vec3f {
    return saturate3((v * (aces_a * v + aces_b)) / (v * (aces_c * v + aces_d) + aces_e));
}

fn tonemapACES4(v: vec4f) -> vec4f {
    return vec4(tonemapACES3(v.rgb), v.a);
}
