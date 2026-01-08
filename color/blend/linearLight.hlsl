#include "linearDodge.hlsl"
#include "linearBurn.hlsl"

/*
contributors: Jamie Owen
description: Photoshop Linear Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendLinearLigth(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDLINEARLIGHT
#define FNC_BLENDLINEARLIGHT
float blendLinearLight(in float base, in float blend) {
  return blend < .5? blendLinearBurn(base, (2. * blend)): blendLinearDodge(base, (2. * (blend- .5)));
}

float3 blendLinearLight(in float3 base, in float3 blend) {
  return float3(blendLinearLight(base.r, blend.r),
                blendLinearLight(base.g, blend.g),
                blendLinearLight(base.b, blend.b));
}

float3 blendLinearLight(in float3 base, in float3 blend, in float opacity) {
    return (blendLinearLight(base, blend) * opacity + base * (1. - opacity));
}
#endif
