/*
contributors: Patricio Gonzalez Vivo
description: power of 3
use: <float|vec2|vec3|vec4> pow3(<float|vec2|vec3|vec4> v)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_POW3
#define FNC_POW3

float pow3(const in float v) { return v * v * v; }
vec2 pow3(const in vec2 v) { return v * v * v; }
vec3 pow3(const in vec3 v) { return v * v * v; }
vec4 pow3(const in vec4 v) { return v * v * v; }

#endif
