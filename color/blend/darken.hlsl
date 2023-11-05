/*
contributors: Jamie Owen
description: Photoshop Darken blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendDarken(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDDARKEN
#define FNC_BLENDDARKEN
float blendDarken(in float base, in float blend) {
    return min(blend,base);
}

float3 blendDarken(in float3 base, in float3 blend) {
    return float3(  blendDarken(base.r, blend.r),
                    blendDarken(base.g, blend.g),
                    blendDarken(base.b, blend.b) );
}

float3 blendDarken(in float3 base, in float3 blend, in float opacity) {
    return (blendDarken(base, blend) * opacity + base * (1. - opacity));
}
#endif
