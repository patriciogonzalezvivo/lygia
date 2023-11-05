/*
contributors: Jamie Owen
description: Photoshop Negation blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendNegation(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDNEGATION
#define FNC_BLENDNEGATION
float blendNegation(in float base, in float blend) {
    return 1. - abs(1. - base - blend);
}

float3 blendNegation(in float3 base, in float3 blend) {
    return float3(1., 1., 1.) - abs(float3(1., 1., 1.) - base - blend);
}

float3 blendNegation(in float3 base, in float3 blend, in float opacity) {
    return (blendNegation(base, blend) * opacity + base * (1. - opacity));
}
#endif
