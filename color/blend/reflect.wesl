/*
contributors: Jamie Owen
description: Photoshop Reflect blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendReflect(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendReflect(base: f32, blend: f32) -> f32 {
  return select(min(base * base / (1.0 - blend), 1.0), blend, blend == 1.0);
}

fn blendReflect3(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    blendReflect(base.r, blend.r),
    blendReflect(base.g, blend.g),
    blendReflect(base.b, blend.b)
  );
}

fn blendReflect3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
  return blendReflect3(base, blend) * opacity + base * (1.0 - opacity);
}
