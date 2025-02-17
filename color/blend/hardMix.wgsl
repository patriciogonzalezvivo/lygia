#include "vividLight.wgsl"

/*
contributors: Jamie Owen
description: Photoshop Hard Mix blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendHardMix(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendHardMix(base: f32, blend: f32) -> f32 {
  return select(1.0, 0.0, blendVividLight(base, blend) < 0.5);
}

fn blendHardMix3(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    blendHardMix(base.r, blend.r),
    blendHardMix(base.g, blend.g),
    blendHardMix(base.b, blend.b)
  );
}

fn blendHardMix3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
  return blendHardMix3(base, blend) * opacity + base * (1.0 - opacity);
}
