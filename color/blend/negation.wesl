/*
contributors: Jamie Owen
description: Photoshop Negation blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendNegation(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendNegation(base: f32, blend: f32) -> f32 {
    return 1.0 - abs(1.0 - base - blend);
}

fn blendNegation3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(1.0) - abs(vec3f(1.0) - base - blend);
}

fn blendNegation3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendNegation3(base, blend) * opacity + base * (1.0 - opacity);
}
