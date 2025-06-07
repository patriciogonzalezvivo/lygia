import lygia::color::blend::reflect::{blendReflect, blendReflect3};

/*
contributors: Jamie Owen
description: Photoshop Glow blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendGlow(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendGlow(base: f32, blend: f32) -> f32 {
  return blendReflect(blend, base);
}

fn blendGlow3(base: vec3f, blend: vec3f) -> vec3f {
  return blendReflect3(blend, base);
}

fn blendGlow3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
  return blendGlow3(base, blend) * opacity + base * (1.0 - opacity);
}
