/*
contributors: Jamie Owen
description: Photoshop Exclusion blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendExclusion(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDEXCLUSION
#define FNC_BLENDEXCLUSION
float blendExclusion(in float base, in float blend) {
    return base + blend - 2. * base * blend;
}

float3 blendExclusion(in float3 base, in float3 blend) {
    return base + blend - 2. * base * blend;
}

float3 blendExclusion(in float3 base, in float3 blend, in float opacity) {
    return (blendExclusion(base, blend) * opacity + base * (1. - opacity));
}
#endif
