import lygia::color::space::{rgb2hsv::rgb2hsv, hsv2rgb::hsv2rgb};

/*
contributors: Romain Dura
description: Hue Blend mode creates the result color by combining the luminance and saturation of the base color with the hue of the blend color.
use: blendHue(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

fn blendHue(base: vec3f, blend: vec3f) -> vec3f {
  let baseHSV = rgb2hsv(base);
  let blendHSV = rgb2hsv(blend);
  return hsv2rgb(vec3f(blendHSV.x, baseHSV.y, baseHSV.z));
}

fn blendHueOpacity(base: vec3f, blend: vec3f, opacity: f32) -> vec3f {
  return blendHue(base, blend) * opacity + base * (1.0 - opacity);
}
