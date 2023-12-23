#include "rgb2hcv.glsl"
#include "hue2rgb.glsl"

/*
contributors: ["David Schaeffer", "tobspr", "Patricio Gonzalez Vivo"]
description: |
    Convert from linear RGB to HCY (Hue, Chroma, Luminance)
    HCY is a cylindrica. From: https://github.com/tobspr/GLSL-Color-Spaces/blob/master/ColorSpaces.inc.glsl
use: <vec3|vec4> rgb2hcy(<vec3|vec4> rgb)
license:
    - MIT License (MIT) Copyright (c) 2015 tobspr
*/

#ifndef HCY_EPSILON
#define HCY_EPSILON 1e-10
#endif

#ifndef FNC_RGB2HCY
#define FNC_RGB2HCY
vec3 rgb2hcy(const in vec3 rgb) {
    const vec3 HCYwts = vec3(0.299, 0.587, 0.114);
    // Corrected by David Schaeffer
    vec3 HCV = rgb2hcv(rgb);
    float Y = dot(rgb, HCYwts);
    float Z = dot(hue2rgb(HCV.x), HCYwts);
    if (Y < Z) {
        HCV.y *= Z / (HCY_EPSILON + Y);
    } else {
        HCV.y *= (1.0 - Z) / (HCY_EPSILON + 1.0 - Y);
    }
    return vec3(HCV.x, HCV.y, Y);
}
vec4 rgb2hcy(vec4 rgb) { return vec4(rgb2hcy(rgb.rgb), rgb.a);}
#endif