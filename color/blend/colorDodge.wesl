/*
contributors: Jamie Owen
description: Photoshop Color Dodge blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendColorDodge(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendColorDodge(base: f32, blend: f32) -> f32 {
  return select(min(base / (1.0 - blend), 1.0), blend, blend == 1.0);
}

fn blendColorDodge3(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    blendColorDodge(base.r, blend.r),
    blendColorDodge(base.g, blend.g),
    blendColorDodge(base.b, blend.b)
  );
}

fn blendColorDodge3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
  return blendColorDodge3(base, blend) * opacity + base * (1.0 - opacity);
}
