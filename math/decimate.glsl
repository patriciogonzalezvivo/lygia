/*
contributors: Patricio Gonzalez Vivo
description: decimate a value with an specific presicion
use: <float|vec2|vec3|vec4> decimate(<float|vec2|vec3|vec4> value, <float|vec2|vec3|vec4> presicion)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_DECIMATE
#define FNC_DECIMATE
float decimate(float v, float p){ return floor(v*p)/p; }
vec2 decimate(vec2 v, float p){ return floor(v*p)/p; }
vec2 decimate(vec2 v, vec2 p){ return floor(v*p)/p; }
vec3 decimate(vec3 v, float p){ return floor(v*p)/p; }
vec3 decimate(vec3 v, vec3 p){ return floor(v*p)/p; }
vec4 decimate(vec4 v, float p){ return floor(v*p)/p; }
vec4 decimate(vec4 v, vec4 p){ return floor(v*p)/p; }
#endif