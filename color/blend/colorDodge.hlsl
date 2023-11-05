/*
contributors: Jamie Owen
description: Photoshop Color Dodge blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendColorDodge(<float|float3> base, <float|float3> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDCOLORDODGE
#define FNC_BLENDCOLORDODGE
float blendColorDodge(in float base, in float blend) {
    return (blend == 1.)? blend: min( base / (1. - blend), 1.);
}

float3 blendColorDodge(in float3 base, in float3 blend) {
    return float3(    blendColorDodge(base.r, blend.r),
                    blendColorDodge(base.g, blend.g),
                    blendColorDodge(base.b, blend.b) );
}

float3 blendColorDodge(in float3 base, in float3 blend, in float opacity) {
    return (blendColorDodge(base, blend) * opacity + base * (1. - opacity));
}
#endif
