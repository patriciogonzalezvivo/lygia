/*
contributors: Jamie Owen
description: Photoshop Soft Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendSubtract(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendSubtract(base: f32, blend: f32) -> f32 {
  return max(base + blend - 1.0, 0.0);
}

fn blendSubtract3(base: vec3f, blend: vec3f) -> vec3f {
  return max(base + blend - vec3f(1.0), vec3f(0.0));
}

fn blendSubtract3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
  return blendSubtract3(base, blend) * opacity + base * (1.0 - opacity);
}
