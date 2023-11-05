#include "vividLight.hlsl"

/*
contributors: Jamie Owen
description: Photoshop Hard Mix blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendHardMix(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDHARDMIX
#define FNC_BLENDHARDMIX
float blendHardMix(in float base, in float blend) {
    return (blendVividLight(base, blend) < .5)? 0.: 1.;
}

float3 blendHardMix(in float3 base, in float3 blend) {
    return float3(  blendHardMix(base.r, blend.r),
                    blendHardMix(base.g, blend.g),
                    blendHardMix(base.b, blend.b) );
}

float3 blendHardMix(in float3 base, in float3 blend, in float opacity) {
    return (blendHardMix(base, blend) * opacity + base * (1. - opacity));
}
#endif
