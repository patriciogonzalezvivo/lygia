/*
contributors: Jamie Owen
description: Photoshop Phoenix blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendPhoenix(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendPhoenix(base: f32, blend: f32) -> f32 {
    return min(base, blend) - max(base, blend) + 1.0;
}

fn blendPhoenix3(base: vec3f, blend: vec3f) -> vec3f {
    return min(base, blend) - max(base, blend) + vec3f(1.0);
}

fn blendPhoenix3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return blendPhoenix3(base, blend) * opacity + base * (1.0 - opacity);
}
