#include "../space/rgb2hsv.hlsl"
#include "../space/hsv2rgb.hlsl"

/*
contributors: Romain Dura
description: Color Blend mode creates the result color by combining the luminance of the base color with the hue and saturation of the blend color.
use: blendColor(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDCOLOR
#define FNC_BLENDCOLOR
float3 blendColor(in float3 base, in float3 blend) {
  float3 baseHSL = rgb2hsv(base);
  float3 blendHSL = rgb2hsv(blend);
  return hsv2rgb(float3(blendHSL.x, blendHSL.y, baseHSL.z));
}

float3 blendColor(in float3 base, in float3 blend, float opacity) { 
    return (blendColor(base, blend) * opacity + base * (1. - opacity)); 
}
#endif
