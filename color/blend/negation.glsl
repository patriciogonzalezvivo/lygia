/*
original_author: Jamie Owen
description: Photoshop Negation blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendNegation(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDNEGATION
#define FNC_BLENDNEGATION
float blendNegation(in float base, in float blend) {
    return 1. - abs(1. - base - blend);
}

vec3 blendNegation(in vec3 base, in vec3 blend) {
    return vec3(1.) - abs(vec3(1.) - base - blend);
}

vec3 blendNegation(in vec3 base, in vec3 blend, in float opacity) {
    return (blendNegation(base, blend) * opacity + base * (1. - opacity));
}
#endif
