/*
contributors: Jamie Owen
description: Photoshop Screen blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendScreen(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendScreen(base: f32, blend: f32) -> f32 {
    return 1. - ((1. - base) * (1. - blend));
}

fn blendScreen3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3f(blendScreen(base.r, blend.r),
                 blendScreen(base.g, blend.g),
                 blendScreen(base.b, blend.b));
}

fn blendScreenWithOpacity3(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return (blendScreen3(base, blend) * opacity + base * (1. - opacity));
}
