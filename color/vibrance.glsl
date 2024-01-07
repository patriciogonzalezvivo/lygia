/*
contributors: Christian Cann Schuldt Jensen ~ CeeJay.dk
description: |
    Vibrance is a smart-tool which cleverly increases the intensity of the more muted colors and leaves the already well-saturated colors alone. Prevents skin tones from becoming overly saturated and unnatural. 
    vibrance from https://github.com/CeeJayDK/SweetFX/blob/master/Shaders/Vibrance.fx 
use: <vec3|vec4> vibrance(<vec3|vec4> color, <float> v) 
license: MIT License (MIT) Copyright (c) 2014 CeeJayDK
*/

#include "../math/mmax.glsl"
#include "../math/mmin.glsl"
#include "luma.glsl"

#ifndef FNC_VIBRANCE
#define FNC_VIBRANCE
vec3 vibrance(in vec3 v, in float vi) {
    float max_v = mmax(v);
    float min_v = mmin(v);
    float sat = max_v - min_v;
    float lum = luma(v);
    return mix(vec3(lum), v, 1.0 + (vi * 1.0 - (sign(vi) * sat)));
}

vec4 vibrance(in vec4 v, in float vi) { return vec4( vibrance(v.rgb, vi), v.a); }
#endif
