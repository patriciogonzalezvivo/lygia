#include "../space/rgb2hsv.glsl"
#include "../space/hsv2rgb.glsl"

/*
contributors: Romain Dura
description: Color Blend mode creates the result color by combining the luminance of the base color with the hue and saturation of the blend color.
use: blendColor(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDCOLOR
#define FNC_BLENDCOLOR
vec3 blendColor(vec3 base, vec3 blend) {
  vec3 baseHSL = rgb2hsv(base);
  vec3 blendHSL = rgb2hsv(blend);
  return hsv2rgb(
      vec3(blendHSL.x, blendHSL.y, baseHSL.z));
}
vec3  blendColor(in vec3 base, in vec3 blend, float opacity) { 
    return (blendColor(base, blend) * opacity + base * (1. - opacity)); 
}
#endif
