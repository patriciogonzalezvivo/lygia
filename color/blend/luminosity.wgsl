#include "../space/rgb2hsv.wgsl"
#include "../space/hsv2rgb.wgsl"

/*
contributors: Romain Dura
description: Luminosity Blend mode creates the result color by combining the hue and luminance of the base color with the saturation of the blend color.
use: blendLuminosity(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendLuminosity(base: vec3f, blend: vec3f) -> vec3f {
  let baseHSL = rgb2hsv(base);
  let blendHSL = rgb2hsv(blend);
  return hsv2rgb(vec3f(baseHSL.x, baseHSL.y, blendHSL.z));
}

fn blendLuminosityOpacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
  return blendLuminosity(base, blend) * opacity + base * (1.0 - opacity);
}
