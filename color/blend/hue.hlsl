#include "../space/rgb2hsv.hlsl"
#include "../space/hsv2rgb.hlsl"

/*
contributors: Romain Dura
description: Hue Blend mode creates the result color by combining the luminance and saturation of the base color with the hue of the blend color.
use: blendHue(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDHUE
#define FNC_BLENDHUE
float3 blendHue(in float3 base, in float3 blend) {
  float3 baseHSL = rgb2hsv(base);
  float3 blendHSL = rgb2hsv(blend);
  return hsv2rgb(float3(blendHSL.x, baseHSL.y, baseHSL.z));
}

float3 blendHue(in float3 base, in float3 blend, float opacity) { 
    return (blendHue(base, blend) * opacity + base * (1. - opacity)); 
}
#endif
