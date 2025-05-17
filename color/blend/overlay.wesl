/*
contributors: Jamie Owen
description: Photoshop Overlay blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendOverlay(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendOverlay(base: f32, blend: f32) -> f32 {
    if (base < 0.5) {
        return (2.*base*blend);
    } else {
        return (1. - 2. * (1. - base) * (1. - blend));
    }
}

fn blendOverlay3(base: vec3f, blend: vec3f) -> vec3f {
    return vec3(blendOverlay(base.r, blend.r),
                blendOverlay(base.g, blend.g),
                blendOverlay(base.b, blend.b));
}

fn blendOverlay3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
    return (blendOverlay3(base, blend) * opacity + base * (1. - opacity));
}