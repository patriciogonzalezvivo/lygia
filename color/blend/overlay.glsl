/*
original_author: Jamie Owen
description: Photoshop Overlay blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendOverlay(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDOVERLAY
#define FNC_BLENDOVERLAY
float blendOverlay(in float base, in float blend) {
    return (base < .5)? (2.*base*blend): (1. - 2. * (1. - base) * (1. - blend));
}

vec3 blendOverlay(in vec3 base, in vec3 blend) {
    return vec3(blendOverlay(base.r, blend.r),
                blendOverlay(base.g, blend.g),
                blendOverlay(base.b, blend.b));
}

vec3 blendOverlay(in vec3 base, in vec3 blend, in float opacity) {
    return (blendOverlay(base, blend) * opacity + base * (1. - opacity));
}
#endif
