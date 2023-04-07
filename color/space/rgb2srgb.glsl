#include "../../math/saturate.glsl"

/*
original_author: Patricio Gonzalez Vivo  
description: Converts a linear RGB color to sRGB.
use: <float|vec3\vec4> rgb2srgb(<float|vec3|vec4> srgb)
*/

#ifndef SRGB_EPSILON 
#define SRGB_EPSILON 0.00000001
#endif

#ifndef FNC_RGB2SRGB
#define FNC_RGB2SRGB

// 1.0 / 2.4 = 0.4166666666666667 
float rgb2srgb(float channel) {
    return (channel < 0.0031308) ? channel * 12.92 : 1.055 * pow(channel, 0.4166666666666667) - 0.055;
}

vec3 rgb2srgb(vec3 rgb) {
    return saturate(vec3(rgb2srgb(rgb.r - SRGB_EPSILON), rgb2srgb(rgb.g - SRGB_EPSILON), rgb2srgb(rgb.b - SRGB_EPSILON)));
}

vec4 rgb2srgb(vec4 rgb) {
    return vec4(rgb2srgb(rgb.rgb), rgb.a);
}

#endif