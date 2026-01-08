/*
contributors: Jamie Owen
description: Photoshop Darken blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendDarken(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendDarken(base: f32, blend: f32) -> f32 {
    return min(blend, base);
}

fn blendDarken3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(
        blendDarken(base.r, blend.r),
        blendDarken(base.g, blend.g),
        blendDarken(base.b, blend.b)
    );
}

fn blendDarken3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendDarken3(base, blend) * opacity + base * (1.0 - opacity);
}
