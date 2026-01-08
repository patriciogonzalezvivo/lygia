/*
contributors: Jamie Owen
description: Photoshop Color Burn blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendColorBurn(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendColorBurn(base: f32, blend: f32) -> f32 {
  return select(max((1.0 - ((1.0 - base) / blend)), 0.0), blend, blend == 0.0);
}

fn blendColorBurn3(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    blendColorBurn(base.r, blend.r),
    blendColorBurn(base.g, blend.g),
    blendColorBurn(base.b, blend.b)
  );
}

fn blendColorBurn3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
  return blendColorBurn3(base, blend) * opacity + base * (1.0 - opacity);
}
