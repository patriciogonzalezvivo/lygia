/*
contributors: Patricio Gonzalez Vivo
description: Change the exposure of a color
use: exposure(<float|vec3|vec4> color, float amount)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_EXPOSURE
#define FNC_EXPOSURE
float exposure(float v, float a) { return v * pow(2., a); }
vec3 exposure(vec3 v, float a) { return v * pow(2., a); }
vec4 exposure(vec4 v, float a) { return vec4(v.rgb * pow(2., a), v.a); }
#endif