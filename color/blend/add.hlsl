/*
contributors: Jamie Owen
description: Photoshop Add blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendAdd(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDADD
#define FNC_BLENDADD
float blendAdd(in float base, in float blend) {
    return min(base + blend, 1.);
}

float3 blendAdd(in float3 base, in float3 blend) {
    return min(base + blend, float3(1., 1., 1.));
}

float3 blendAdd(in float3 base, in float3 blend, float opacity) {
    return (blendAdd(base, blend) * opacity + base * (1. - opacity));
}
#endif
