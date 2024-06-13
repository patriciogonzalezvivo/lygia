/*
contributors: Johan Ismael
description: |
    Color input range adjustment similar to Levels adjusment tool in Photoshop
    Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levelsInputRange(<vec3|vec4> color, <float|vec3> minInput, <float|vec3> maxInput)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LEVELSINPUTRANGE
#define FNC_LEVELSINPUTRANGE
vec3 levelsInputRange(in vec3 v, in vec3 iMin, in vec3 iMax) { return min(max(v - iMin, vec3(0.)) / (iMax - iMin), vec3(1.)); }
vec4 levelsInputRange(in vec4 v, in vec3 iMin, in vec3 iMax) { return vec4(levelsInputRange(v.rgb, iMin, iMax), v.a); }
vec3 levelsInputRange(in vec3 v, in float iMin, in float iMax) { return levelsInputRange(v, vec3(iMin), vec3(iMax)); }
vec4 levelsInputRange(in vec4 v, in float iMin, in float iMax) { return vec4(levelsInputRange(v.rgb, iMin, iMax), v.a); }
#endif