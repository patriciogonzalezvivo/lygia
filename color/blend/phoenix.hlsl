/*
contributors: Jamie Owen
description: Photoshop Phoenix blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendPhoenix(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDPHOENIX
#define FNC_BLENDPHOENIX
float blendPhoenix(in float base, in float blend) {
    return min(base, blend) - max(base, blend) + 1.;
}

float3 blendPhoenix(in float3 base, in float3 blend) {
    return min(base, blend) - max(base, blend) + float3(1., 1., 1.);
}

float3 blendPhoenix(in float3 base, in float3 blend, in float opacity) {
    return (blendPhoenix(base, blend) * opacity + base * (1. - opacity));
}
#endif
