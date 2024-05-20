#include "overlay.glsl"

/*
contributors: Jamie Owen
description: Photoshop HardLight blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendHardLight(<float|vec3> base, <float|vec3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDHARDLIGHT
#define FNC_BLENDHARDLIGHT
float blendHardLight(in float base, in float blend) {
    return blendOverlay(blend, base);
}

vec3 blendHardLight(in vec3 base, in vec3 blend) {
    return blendOverlay(blend, base);
}

vec3 blendHardLight(in vec3 base, in vec3 blend, in float opacity) {
    return (blendHardLight(base, blend) * opacity + base * (1. - opacity));
}
#endif
