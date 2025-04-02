#include "overlay.wgsl"

/*
contributors: Jamie Owen
description: Photoshop HardLight blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendHardLight(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendHardLight(base: f32, blend: f32) -> f32 {
  return blendOverlay(blend, base);
}

fn blendHardLight3(base: vec3f, blend: vec3f) -> vec3f {
  return blendOverlay(blend, base);
}

fn blendHardLight3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
  return blendHardLight3(base, blend) * opacity + base * (1.0 - opacity);
}
