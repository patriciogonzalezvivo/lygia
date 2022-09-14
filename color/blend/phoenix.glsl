/*
original_author: Jamie Owen
description: Photoshop Phoenix blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendPhoenix(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDPHOENIX
#define FNC_BLENDPHOENIX
float blendPhoenix(in float base, in float blend) {
    return min(base, blend) - max(base, blend) + 1.;
}

vec3 blendPhoenix(in vec3 base, in vec3 blend) {
    return min(base, blend) - max(base, blend) + vec3(1.);
}

vec3 blendPhoenix(in vec3 base, in vec3 blend, in float opacity) {
    return (blendPhoenix(base, blend) * opacity + base * (1. - opacity));
}
#endif
