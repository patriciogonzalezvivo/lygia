#include "reflect.hlsl"

/*
contributors: Jamie Owen
description: Photoshop Glow blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendGlow(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDGLOW
#define FNC_BLENDGLOW
float blendGlow(in float base, in float blend) {
    return blendReflect(blend, base);
}

float3 blendGlow(in float3 base, in float3 blend) {
    return blendReflect(blend, base);
}

float3 blendGlow(in float3 base, in float3 blend, in float opacity) {
    return (blendGlow(base, blend) * opacity + base * (1. - opacity));
}
#endif
