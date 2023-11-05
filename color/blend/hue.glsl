#include "../space/rgb2hsv.glsl"
#include "../space/hsv2rgb.glsl"

/*
contributors: Romain Dura
description: Hue Blend mode creates the result color by combining the luminance and saturation of the base color with the hue of the blend color.
use: blendHue(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDHUE
#define FNC_BLENDHUE
vec3 blendHue(vec3 base, vec3 blend) {
  vec3 baseHSL = rgb2hsv(base);
  vec3 blendHSL = rgb2hsv(blend);
  return hsv2rgb(
      vec3(blendHSL.x, baseHSL.y, baseHSL.z));
}
vec3  blendHue(in vec3 base, in vec3 blend, float opacity) { 
    return (blendHue(base, blend) * opacity + base * (1. - opacity)); 
}
#endif
