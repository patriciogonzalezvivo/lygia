/*
contributors: Jamie Owen
description: Photoshop Reflect blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendReflect(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDREFLECT
#define FNC_BLENDREFLECT
float blendReflect(in float base, in float blend) {
    return (blend == 1.)? blend : min(base * base / (1. - blend), 1.);
}

float3 blendReflect(in float3 base, in float3 blend) {
    return float3(  blendReflect(base.r, blend.r),
                    blendReflect(base.g, blend.g),
                    blendReflect(base.b, blend.b) );
}

float3 blendReflect(in float3 base, in float3 blend, in float opacity) {
    return (blendReflect(base, blend) * opacity + base * (1. - opacity));
}
#endif
