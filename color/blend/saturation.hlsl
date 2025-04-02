#include "../space/rgb2hsv.hlsl"
#include "../space/hsv2rgb.hlsl"

/*
contributors: Romain Dura
description: Saturation Blend mode creates the result color by combining the hue and luminance of the base color with the saturation of the blend color.
use: blendSaturation(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDSATURATION
#define FNC_BLENDSATURATION
float3 blendSaturation(in float3 base, in float3 blend) {
  float3 baseHSL = rgb2hsv(base);
  float3 blendHSL = rgb2hsv(blend);
  return hsv2rgb(float3(baseHSL.x, blendHSL.y, baseHSL.z));
}

float3 blendSaturation(in float3 base, in float3 blend, float opacity) { 
    return (blendSaturation(base, blend) * opacity + base * (1. - opacity)); 
}
#endif
