#include "levels/inputRange.glsl"
#include "levels/outputRange.glsl"
#include "levels/gamma.glsl"

/*
contributors: Johan Ismael
description: |
    Combines inputRange, outputRange and gamma functions into one
    Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levels(<vec3|vec4> color, <float|vec3> minInput, <float|vec3> gamma, <float|vec3 maxInput, <float|vec3 minOutput, <float|vec3 maxOutput)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LEVELS
#define FNC_LEVELS
vec3 levels(in vec3 v, in vec3 iMin, in vec3 g, in vec3 iMax, in vec3 oMin, in vec3 oMax) {
    return levelsOutputRange( levelsGamma( levelsInputRange(v, iMin, iMax), g), oMin, oMax);
}

vec3 levels(in vec3 v, in float iMin, in float g, in float iMax, in float oMin, in float oMax) {
    return levels(v, vec3(iMin), vec3(g), vec3(iMax), vec3(oMin), vec3(oMax));
}

vec4 levels(in vec4 v, in vec3 iMin, in vec3 g, in vec3 iMax, in vec3 oMin, in vec3 oMax) {
    return vec4(levels(v.rgb, iMin, g, iMax, oMin, oMax), v.a);
}

vec4 levels(in vec4 v, in float iMin, in float g, in float iMax, in float oMin, in float oMax) {
    return vec4(levels(v.rgb, iMin, g, iMax, oMin, oMax), v.a);
}
#endif