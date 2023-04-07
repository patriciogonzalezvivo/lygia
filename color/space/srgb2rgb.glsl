/*
original_author: Patricio Gonzalez Vivo  
description: sRGB to linear RGB conversion.
use: <float|vec3\vec4> srgb2rgb(<float|vec3|vec4> srgb)
*/

#ifndef SRGB_EPSILON 
#define SRGB_EPSILON 0.00000001
#endif

#ifndef FNC_SRGB2RGB
#define FNC_SRGB2RGB

// 1.0 / 12.92 = 0.0773993808
// 1.0 / (1.0 + 0.055) = 0.947867298578199

float srgb2rgb(float channel) {
    return (channel < 0.04045) ? channel * 0.0773993808 : pow((channel + 0.055) * 0.947867298578199, 2.4);
}

vec3 srgb2rgb(vec3 srgb) {
    return vec3(srgb2rgb(srgb.r + SRGB_EPSILON), 
                srgb2rgb(srgb.g + SRGB_EPSILON), \
                srgb2rgb(srgb.b + SRGB_EPSILON));
}

vec4 srgb2rgb(vec4 srgb) {
    return vec4(srgb2rgb(srgb.rgb), srgb.a);
}

#endif