#include "../../math/saturate.glsl"
#include "hue2rgb.glsl"
/*
contributors: Inigo Quiles
description: |
    Convert from HSV to linear RGB
use: <vec3|vec4> hsv2rgb(<vec3|vec4> hsv)
*/

#ifndef FNC_HSV2RGB
#define FNC_HSV2RGB
vec3 hsv2rgb(const in vec3 hsv) { return ((hue2rgb(hsv.x) - 1.0) * hsv.y + 1.0) * hsv.z; }
vec4 hsv2rgb(const in vec4 hsv) { return vec4(hsv2rgb(hsv.rgb), hsv.a); }
#endif