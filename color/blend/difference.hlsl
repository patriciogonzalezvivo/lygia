/*
contributors: Jamie Owen
description: Photoshop Difference blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendDifference(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDDIFFERENCE
#define FNC_BLENDDIFFERENCE
float blendDifference(in float base, in float blend) {
    return abs(base-blend);
}

float3 blendDifference(in float3 base, in float3 blend) {
    return abs(base-blend);
}

float3 blendDifference(in float3 base, in float3 blend, in float opacity) {
    return (blendDifference(base, blend) * opacity + base * (1. - opacity));
}
#endif
