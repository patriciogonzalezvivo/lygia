/*
contributors: Jamie Owen
description: Photoshop Soft Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendSoftLight(<float|float3|float4> base, <float|float3|float4> blend [, <float> opacity])
license: MIT License (MIT) Copyright (c) 2015 Jamie Owen
*/

#ifndef FNC_BLENDSOFTLIGHT
#define FNC_BLENDSOFTLIGHT
float blendSoftLight(in float base, in float blend) {
    return (blend < .5)? (2. * base * blend + base * base * (1. - 2.*blend)): (sqrt(base) * (2. * blend - 1.) + 2. * base * (1. - blend));
}

float3 blendSoftLight(in float3 base, in float3 blend) {
    return float3(  blendSoftLight(base.r, blend.r),
                    blendSoftLight(base.g, blend.g),
                    blendSoftLight(base.b, blend.b) );
}

float4 blendSoftLight(in float4 base, in float4 blend) {
    return float4(  blendSoftLight( base.r, blend.r ),
                    blendSoftLight( base.g, blend.g ),
                    blendSoftLight( base.b, blend.b ),
                    blendSoftLight( base.a, blend.a )
    );
}

float3 blendSoftLight(in float3 base, in float3 blend, in float opacity) {
    return (blendSoftLight(base, blend) * opacity + base * (1. - opacity));
}
#endif
