/*
contributors: Jamie Owen
description: Photoshop Lighten blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendLighten(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendLighten(base: f32, blend: f32) -> f32 {
  return max(blend, base);
}

fn blendLighten3(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    blendLighten(base.r, blend.r),
    blendLighten(base.g, blend.g),
    blendLighten(base.b, blend.b)
  );
}

fn blendLighten3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
  return blendLighten3(base, blend) * opacity + base * (1.0 - opacity);
}
