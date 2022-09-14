/*
original_author: Jamie Owen
description: Photoshop Soft Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendSoftLight(<float|vec3> base, <float|vec3> blend [, <float> opacity])
*/

#ifndef FNC_BLENDSOFTLIGHT
#define FNC_BLENDSOFTLIGHT
float blendSoftLight(in float base, in float blend) {
    return (blend < .5)? (2. * base * blend + base * base * (1. - 2.*blend)): (sqrt(base) * (2. * blend - 1.) + 2. * base * (1. - blend));
}

vec3 blendSoftLight(in vec3 base, in vec3 blend) {
    return vec3(blendSoftLight(base.r, blend.r),
                blendSoftLight(base.g, blend.g),
                blendSoftLight(base.b, blend.b));
}

vec4 blendSoftLight(in vec4 base, in vec4 blend) {
    return vec4(blendSoftLight( base.r, blend.r ),
                blendSoftLight( base.g, blend.g ),
                blendSoftLight( base.b, blend.b ),
                blendSoftLight( base.a, blend.a )
    );
}

vec3 blendSoftLight(in vec3 base, in vec3 blend, in float opacity) {
    return (blendSoftLight(base, blend) * opacity + base * (1. - opacity));
}
#endif
