/*
contributors: Jamie Owen
description: Photoshop Linear Dodge blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendLinearDodge(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendLinearDodge(base: f32, blend: f32) -> f32 {
    // Note: Same implementation as BlendAddf
    return min(base + blend, 1.0);
}

fn blendLinearDodge3(base: vec3f, blend: vec3f) -> vec3f {
    // Note: Same implementation as BlendAdd
    return min(base + blend, vec3f(1.0));
}

fn blendLinearDodge3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendLinearDodge3(base, blend) * opacity + base * (1.0 - opacity);
}
