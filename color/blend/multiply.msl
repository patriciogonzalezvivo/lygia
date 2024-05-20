/*
contributors: Jamie Owen
description: Photoshop Multiply blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendMultiply(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDMULTIPLY
#define FNC_BLENDMULTIPLY
float blendMultiply(in float base, in float blend) {
    return base * blend;
}

vec3 blendMultiply(in vec3 base, in vec3 blend) {
    return base * blend;
}

vec3 blendMultiply(in vec3 base, in vec3 blend, float opacity) {
    return (blendMultiply(base, blend) * opacity + base * (1. - opacity));
}
#endif
