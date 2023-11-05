/*
contributors: Jamie Owen
description: Photoshop Linear Dodge blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendLinearDodge(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDLINEARDODGE
#define FNC_BLENDLINEARDODGE
float blendLinearDodge(in float base, in float blend) {
  // Note : Same implementation as BlendAddf
    return min(base + blend, 1.);
}

float3 blendLinearDodge(in float3 base, in float3 blend) {
  // Note : Same implementation as BlendAdd
    return min(base + blend, float3(1., 1., 1.));
}

float3 blendLinearDodge(in float3 base, in float3 blend, in float opacity) {
    return (blendLinearDodge(base, blend) * opacity + base * (1. - opacity));
}
#endif
