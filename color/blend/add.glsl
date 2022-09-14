/*
original_author: Jamie Owen
description: Photoshop Add blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendAdd(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDADD
#define FNC_BLENDADD
float blendAdd(in float base, in float blend) {
    return min(base + blend, 1.);
}

vec3 blendAdd(in vec3 base, in vec3 blend) {
    return min(base + blend, vec3(1.));
}

vec3 blendAdd(in vec3 base, in vec3 blend, float opacity) {
    return (blendAdd(base, blend) * opacity + base * (1. - opacity));
}
#endif
