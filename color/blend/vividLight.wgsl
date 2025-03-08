#include "colorBurn.wgsl"
#include "colorDodge.wgsl"

/*
contributors: Jamie Owen
description: Photoshop Vivid Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendVividLight(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendVividLight(base: f32, blend: f32) -> f32 {
  return select(blendColorDodge(base, (blend - 0.5) * 2.0), blendColorBurn(base, blend * 2.0), blend < 0.5);
}

fn blendVividLight3(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    blendVividLight(base.r, blend.r),
    blendVividLight(base.g, blend.g),
    blendVividLight(base.b, blend.b)
  );
}

fn blendVividLight3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
  return blendVividLight3(base, blend) * opacity + base * (1.0 - opacity);
}
