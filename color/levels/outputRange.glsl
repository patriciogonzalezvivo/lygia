/*
contributors: Johan Ismael
description: |
    Color output range adjustment similar to Levels adjustment in Photoshop
    Adapted from Romain Dura (http://mouaif.wordpress.com/?p=94)
use: levelsOutputRange(<vec3|vec4> color, float minOutput, float maxOutput)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LEVELSOUTPUTRANGE
#define FNC_LEVELSOUTPUTRANGE
vec3 levelsOutputRange(in vec3 v, in vec3 oMin, in vec3 oMax) { return mix(oMin, oMax, v); }
vec4 levelsOutputRange(in vec4 v, in vec3 oMin, in vec3 oMax) { return vec4(levelsOutputRange(v.rgb, oMin, oMax), v.a); }
vec3 levelsOutputRange(in vec3 v, in float oMin, in float oMax) { return levelsOutputRange(v, vec3(oMin), vec3(oMax)); }
vec4 levelsOutputRange(in vec4 v, in float oMin, in float oMax) { return vec4(levelsOutputRange(v.rgb, oMin, oMax), v.a); }
#endif