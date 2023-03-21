/*
original_author: Christian Cann Schuldt Jensen ~ CeeJay.dk
description: |
    Vibrance is a smart-tool which cleverly increases the intensity of the more muted colors and leaves the already well-saturated colors alone. Prevents skin tones from becoming overly saturated and unnatural. 
    vibrance from https://github.com/CeeJayDK/SweetFX/blob/master/Shaders/Vibrance.fx 
use: <vec3|vec4> vibrance(<vec3|vec4> color, <float> v) 
*/

#include "../math/mmax.glsl"
#include "../math/mmin.glsl"
#include "luma.glsl"

#ifndef FNC_VIBRANCE
#define FNC_VIBRANCE

vec3 vibrance(in vec3 color, in float v) {
    float max_color = mmax(color);
    float min_color = mmin(color);
    float sat = max_color - min_color;
    float lum = luma(color);
    return mix(vec3(lum), color, 1.0 + (v * 1.0 - (sign(v) * sat)));
}

vec4 vibrance(in vec4 color, in float v) { return vec4( vibrance(color.rgb, v), color.a); }
#endif