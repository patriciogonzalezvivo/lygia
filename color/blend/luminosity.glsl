#include "../space/rgb2hsv.glsl"
#include "../space/hsv2rgb.glsl"

/*
contributors: Romain Dura
description: Luminosity Blend mode creates the result color by combining the hue and luminance of the base color with the saturation of the blend color.
use: blendLuminosity(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDLUMINOSITY
#define FNC_BLENDLUMINOSITY
vec3 blendLuminosity(vec3 base, vec3 blend) {
  vec3 baseHSL = rgb2hsv(base);
  vec3 blendHSL = rgb2hsv(blend);
  return hsv2rgb(
      vec3(baseHSL.x, baseHSL.y, blendHSL.z));
}
vec3  blendLuminosity(in vec3 base, in vec3 blend, float opacity) { 
    return (blendLuminosity(base, blend) * opacity + base * (1. - opacity)); 
}
#endif
