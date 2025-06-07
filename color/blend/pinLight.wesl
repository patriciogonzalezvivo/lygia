import lygia::color::blend::lighten::blendLighten;
import lygia::color::blend::darken::blendDarken;

/*
contributors: Jamie Owen
description: Photoshop Pin Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendPinLight(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendPinLight(base: f32, blend: f32) -> f32 {
  return select(blendLighten(base, (blend - 0.5) * 2.0), blendDarken(base, blend * 2.0), blend < 0.5);
}

fn blendPinLight3(base: vec3f, blend: vec3f) -> vec3f {
  return vec3f(
    blendPinLight(base.r, blend.r),
    blendPinLight(base.g, blend.g),
    blendPinLight(base.b, blend.b)
  );
}

fn blendPinLight3Opacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
  return blendPinLight3(base, blend) * opacity + base * (1.0 - opacity);
}
