/*
contributors: Jamie Owen
description: Photoshop Soft Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendSubtract(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDSUBTRACT
#define FNC_BLENDSUBTRACT
float blendSubtract(in float base, in float blend) {
    return max(base + blend - 1., 0.);
}

float3 blendSubtract(in float3 base, in float3 blend) {
    return max(base + blend - float3(1., 1., 1.), float3(0., 0., 0.));
}

float3 blendSubtract(in float3 base, in float3 blend, in float opacity) {
    return (blendSubtract(base, blend) * opacity + base * (1. - opacity));
}
#endif
