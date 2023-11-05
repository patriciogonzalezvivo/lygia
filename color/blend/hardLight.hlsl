#include "overlay.hlsl"

/*
contributors: Jamie Owen
description: Photoshop HardLight blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendHardLight(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDHARDLIGHT
#define FNC_BLENDHARDLIGHT
float blendHardLight(in float base, in float blend) {
    return blendOverlay(blend, base);
}

float3 blendHardLight(in float3 base, in float3 blend) {
    return blendOverlay(blend, base);
}

float3 blendHardLight(in float3 base, in float3 blend, in float opacity) {
    return (blendHardLight(base, blend) * opacity + base * (1. - opacity));
}
#endif
